#pragma once

#include "core/ActionResponse.hpp"
#include "core/concepts/GameConcept.hpp"
#include "search/VerboseDataBase.hpp"

#include <memory>

namespace search {

template <core::concepts::Game Game>
struct AuxData {
  using ActionResponse = core::ActionResponse<Game>;
  using VerboseDataBase_sptr = std::shared_ptr<generic::VerboseDataBase>;

  ActionResponse action_response;
  VerboseDataBase_sptr verbose_data;
};

}  // namespace search
