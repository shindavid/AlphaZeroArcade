#pragma once

# include "search/VerboseDataBase.hpp"

namespace generic {

struct VerboseManager {
  static VerboseManager* get_instance() {
    static VerboseManager instance;
    return &instance;
  }

  void set(VerboseDataBase* verbose_data) { verbose_data_ = verbose_data; }
  VerboseDataBase* verbose_data() const { return verbose_data_; }

  private:
    VerboseDataBase* verbose_data_ = nullptr;
};

}  // namespace generic
