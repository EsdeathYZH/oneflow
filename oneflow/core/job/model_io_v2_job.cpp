/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/job/model_io_v2_job.h"
#include "oneflow/core/operator/interface_op_util.h"
#include "oneflow/core/common/buffer_manager.h"

namespace oneflow {

namespace {

bool CompareVariableOpConf(const VariableOpConf& lhs, const VariableOpConf& rhs) {
  if (lhs.has_random_seed() && rhs.has_random_seed()) {
    CHECK_EQ(lhs.random_seed(), rhs.random_seed());
  }
  VariableOpConf var_conf_a(lhs);
  VariableOpConf var_conf_b(rhs);
  var_conf_a.clear_tick();
  var_conf_b.clear_tick();
  var_conf_a.clear_out();
  var_conf_b.clear_out();
  var_conf_a.clear_trainable();
  var_conf_b.clear_trainable();
  return PbMd::Equals(var_conf_a, var_conf_b);
}

OperatorConf GenForeignInputOpConf(const std::string& job_name, const int64_t input_size) {
  OperatorConf foreign_input_op_conf{};
  foreign_input_op_conf.set_name("System-Push-ForeignInput_" + NewUniqueId());
  ForeignInputOpConf* foreign_input_conf = foreign_input_op_conf.mutable_foreign_input_conf();
  foreign_input_conf->set_out("out");
  foreign_input_conf->set_ofblob_buffer_name(GetForeignInputBufferName(job_name));
  InterfaceBlobConf* blob_conf = foreign_input_conf->mutable_blob_conf();
  *blob_conf->mutable_shape()->mutable_dim()->Add() = input_size;
  blob_conf->set_is_dynamic(true);
  blob_conf->set_data_type(DataType::kInt8);
  return foreign_input_op_conf;
}

void SetModelIoDefaultJobConf(JobConfigProto* job_conf, const std::string& job_name) {
  job_conf->set_job_name(job_name);
  job_conf->mutable_predict_conf();
}

OperatorConf GenTickOpConf(const std::string& op_name) {
  OperatorConf tick_op_conf{};
  tick_op_conf.set_name(op_name);
  tick_op_conf.mutable_tick_conf()->set_out("out");
  return tick_op_conf;
}

OperatorConf CloneVariableOpConf(const OperatorConf& variable_op_conf) {
  OperatorConf new_var_op_conf(variable_op_conf);
  new_var_op_conf.mutable_variable_conf()->clear_tick();
  return new_var_op_conf;
}

std::string GetVariableLbn(const OperatorConf& variable_op_conf) {
  return GenLogicalBlobName(variable_op_conf.name(), variable_op_conf.variable_conf().out());
}

void FilterVariableOps(const std::vector<std::shared_ptr<Job>>& jobs,
                       HashMap<std::string, OperatorConf>* var_op_name2op_conf) {
  FOR_RANGE(int64_t, job_id, 0, jobs.size()) {
    for (const OperatorConf& op_conf : jobs.at(job_id)->net().op()) {
      if (op_conf.has_variable_conf()) {
        if (var_op_name2op_conf->find(op_conf.name()) == var_op_name2op_conf->end()) {
          CHECK(var_op_name2op_conf->emplace(op_conf.name(), op_conf).second);
        } else {
          CHECK(CompareVariableOpConf(var_op_name2op_conf->at(op_conf.name()).variable_conf(),
                                      op_conf.variable_conf()));
        }
      }
    }
  }
}

void MakeModelInitJob(
    const std::string& job_name, Job* job,
    const HashMap<std::string, OperatorConf>& var_op_name2op_conf,
    const HashMap<std::string, ParallelBlobConf>& var_op_name2parallel_blob_conf) {
  auto* flag_name2flag_value = job->mutable_job_conf()->mutable_flag_name2flag_value();
  (*flag_name2flag_value)["__is_user_function__"].set_at_bool(false);
  SetModelIoDefaultJobConf(job->mutable_job_conf(), job_name);
  Global<InterUserJobInfo>::Get()->set_global_model_init_job_name(job_name);
  JobBuilder job_builder(job);
  const ParallelConf master_parallel_conf = GenParallelConfOfCpuZeroOnMaster();
  const OperatorConf tick_op_conf = GenTickOpConf("System-ModelInit-Tick");
  const OperatorConf foreign_input_op_conf = GenForeignInputOpConf(job_name, 1);
  job_builder.AddOps(master_parallel_conf, {foreign_input_op_conf, tick_op_conf});
  if (var_op_name2op_conf.empty()) { return; }
  HashMap<ParallelConf, std::vector<OperatorConf>> parallel_conf2variable_op_conf;
  for (const auto& pair : var_op_name2op_conf) {
    const auto& var_op_name = pair.first;
    const OperatorConf& variable_op_conf = pair.second;
    const ParallelBlobConf& parallel_blob_conf = var_op_name2parallel_blob_conf.at(var_op_name);
    parallel_conf2variable_op_conf[parallel_blob_conf.parallel_conf()].emplace_back(
        variable_op_conf);
    OperatorConf new_var_op_conf = CloneVariableOpConf(variable_op_conf);
    job_builder.AddOps(parallel_blob_conf.parallel_conf(), {new_var_op_conf});
  }
  for (auto& pair : parallel_conf2variable_op_conf) {
    std::vector<OperatorConf>& variable_op_confs = pair.second;
    OperatorConf model_init_op_conf{};
    model_init_op_conf.set_name("System-ModelInit-" + NewUniqueId());
    ModelInitV2OpConf* model_init_conf = model_init_op_conf.mutable_model_init_v2_conf();
    const int64_t num_var = variable_op_confs.size();
    model_init_conf->mutable_ref()->Reserve(num_var);
    model_init_conf->mutable_variable_op_name()->Reserve(num_var);
    model_init_conf->mutable_original_variable_conf()->Reserve(num_var);
    for (int64_t i = 0; i < num_var; ++i) {
      model_init_conf->add_ref(GetVariableLbn(variable_op_confs.at(i)));
      model_init_conf->add_variable_op_name(variable_op_confs.at(i).name());
      *model_init_conf->add_original_variable_conf() =
          std::move(*variable_op_confs.at(i).mutable_variable_conf());
    }
    job_builder.AddOps(pair.first, {model_init_op_conf});
  }
}

void MakeModelLoadJob(
    const std::string& job_name, Job* job,
    const HashMap<std::string, OperatorConf>& var_op_name2op_conf,
    const HashMap<std::string, ParallelBlobConf>& var_op_name2parallel_blob_conf) {
  auto* flag_name2flag_value = job->mutable_job_conf()->mutable_flag_name2flag_value();
  (*flag_name2flag_value)["__is_user_function__"].set_at_bool(false);
  SetModelIoDefaultJobConf(job->mutable_job_conf(), job_name);
  Global<InterUserJobInfo>::Get()->set_global_model_load_job_name(job_name);
  JobBuilder job_builder(job);
  const ParallelConf master_parallel_conf = GenParallelConfOfCpuZeroOnMaster();
  const OperatorConf tick_op_conf = GenTickOpConf("System-ModelLoad-Tick");
  const OperatorConf foreign_input_op_conf = GenForeignInputOpConf(job_name, 65536);
  job_builder.AddOps(master_parallel_conf, {foreign_input_op_conf, tick_op_conf});
  if (var_op_name2op_conf.empty()) { return; }
  HashMap<ParallelConf, std::vector<OperatorConf>> parallel_conf2variable_op_conf;
  for (const auto& pair : var_op_name2op_conf) {
    const auto& var_op_name = pair.first;
    const OperatorConf& variable_op_conf = pair.second;
    const ParallelBlobConf& parallel_blob_conf = var_op_name2parallel_blob_conf.at(var_op_name);
    parallel_conf2variable_op_conf[parallel_blob_conf.parallel_conf()].emplace_back(
        variable_op_conf);
    OperatorConf new_var_op_conf = CloneVariableOpConf(variable_op_conf);
    job_builder.AddOps(parallel_blob_conf.parallel_conf(), {new_var_op_conf});
  }
  for (auto& pair : parallel_conf2variable_op_conf) {
    std::vector<OperatorConf>& variable_op_confs = pair.second;
    OperatorConf model_load_op_conf{};
    model_load_op_conf.set_name("System-ModelLoad-" + NewUniqueId());
    ModelLoadV2OpConf* model_load_conf = model_load_op_conf.mutable_model_load_v2_conf();
    model_load_conf->set_path(GenLogicalBlobName(foreign_input_op_conf.name(),
                                                 foreign_input_op_conf.foreign_input_conf().out()));
    const int64_t num_var = variable_op_confs.size();
    model_load_conf->mutable_ref()->Reserve(num_var);
    model_load_conf->mutable_variable_op_name()->Reserve(num_var);
    model_load_conf->mutable_original_variable_conf()->Reserve(num_var);
    for (int64_t i = 0; i < num_var; ++i) {
      model_load_conf->add_ref(GetVariableLbn(variable_op_confs.at(i)));
      model_load_conf->add_variable_op_name(variable_op_confs.at(i).name());
      *model_load_conf->add_original_variable_conf() =
          std::move(*variable_op_confs.at(i).mutable_variable_conf());
    }
    job_builder.AddOps(pair.first, {model_load_op_conf});
  }
}

void MakeModelSaveJob(
    const std::string& job_name, Job* job,
    const HashMap<std::string, OperatorConf>& var_op_name2op_conf,
    const HashMap<std::string, ParallelBlobConf>& var_op_name2parallel_blob_conf) {
  auto* flag_name2flag_value = job->mutable_job_conf()->mutable_flag_name2flag_value();
  (*flag_name2flag_value)["__is_user_function__"].set_at_bool(false);
  Global<InterUserJobInfo>::Get()->set_global_model_save_job_name(job_name);
  SetModelIoDefaultJobConf(job->mutable_job_conf(), job_name);
  JobBuilder job_builder(job);
  ParallelConf master_parallel_conf = GenParallelConfOfCpuZeroOnMaster();
  const OperatorConf tick_op_conf = GenTickOpConf("System-ModelSave-Tick");
  const OperatorConf foreign_input_op_conf = GenForeignInputOpConf(job_name, 65536);
  job_builder.AddOps(master_parallel_conf, {foreign_input_op_conf, tick_op_conf});
  if (var_op_name2op_conf.empty()) { return; }
  HashMap<ParallelConf, std::vector<OperatorConf>> parallel_conf2variable_op_conf;
  for (const auto& pair : var_op_name2op_conf) {
    const auto& var_op_name = pair.first;
    const OperatorConf& variable_op_conf = pair.second;
    const auto& parallel_blob_conf = var_op_name2parallel_blob_conf.at(var_op_name);
    parallel_conf2variable_op_conf[parallel_blob_conf.parallel_conf()].emplace_back(
        variable_op_conf);
    OperatorConf new_var_op_conf = CloneVariableOpConf(variable_op_conf);
    job_builder.AddOps(parallel_blob_conf.parallel_conf(), {new_var_op_conf});
  }
  for (auto pair : parallel_conf2variable_op_conf) {
    std::vector<OperatorConf>& variable_op_confs = pair.second;
    OperatorConf model_save_op_conf{};
    model_save_op_conf.set_name("System-ModelSave-" + NewUniqueId());
    ModelSaveV2OpConf* model_save_conf = model_save_op_conf.mutable_model_save_v2_conf();
    model_save_conf->set_path(GenLogicalBlobName(foreign_input_op_conf.name(),
                                                 foreign_input_op_conf.foreign_input_conf().out()));
    const int64_t num_var = variable_op_confs.size();
    model_save_conf->mutable_in()->Reserve(num_var);
    model_save_conf->mutable_variable_op_name()->Reserve(num_var);
    model_save_conf->mutable_original_variable_conf()->Reserve(num_var);
    for (int64_t i = 0; i < num_var; ++i) {
      *model_save_conf->add_in() = GetVariableLbn(variable_op_confs.at(i));
      *model_save_conf->add_variable_op_name() = variable_op_confs.at(i).name();
      *model_save_conf->add_original_variable_conf() =
          std::move(*variable_op_confs.at(i).mutable_variable_conf());
    }
    job_builder.AddOps(pair.first, {model_save_op_conf});
  }
}

}  // namespace

void MakeModelIoV2Jobs(const std::vector<std::shared_ptr<Job>>& jobs,
                       const HashMap<std::string, ParallelBlobConf>& var_op_name2parallel_blob_conf,
                       const std::function<void(Job*)>& Handler) {
  HashMap<std::string, OperatorConf> var_op_name2op_conf;
  FilterVariableOps(jobs, &var_op_name2op_conf);
  {
    Job model_init_job;
    MakeModelInitJob("System-ModelInit", &model_init_job, var_op_name2op_conf,
                     var_op_name2parallel_blob_conf);
    Handler(&model_init_job);
  }
  {
    Job model_load_job;
    MakeModelLoadJob("System-ModelLoad", &model_load_job, var_op_name2op_conf,
                     var_op_name2parallel_blob_conf);
    Handler(&model_load_job);
  }
  {
    Job model_save_job;
    MakeModelSaveJob("System-ModelSave", &model_save_job, var_op_name2op_conf,
                     var_op_name2parallel_blob_conf);
    Handler(&model_save_job);
  }
}

}  // namespace oneflow
