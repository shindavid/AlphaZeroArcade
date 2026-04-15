#pragma once

#include "core/AbstractPlayerGenerator.hpp"
#include "core/OraclePool.hpp"
#include "core/PlayerFactory.hpp"
#include "games/chess/Game.hpp"
#include "games/chess/UciProcess.hpp"

namespace a0achess {

template <typename PlayerT>
class UciPlayerGenerator : public core::AbstractPlayerGenerator<Game> {
 public:
  using UciPool = core::OraclePool<UciProcess>;
  using Player = core::AbstractPlayer<Game>;
  using PlayerParams = PlayerT::Params;

  UciPlayerGenerator(UciPool& pool) : pool_(pool) {}

  std::string get_default_name() const override { return PlayerT::default_name(params_); }
  std::string type_str() const override { return std::string(PlayerT::kTypeStr); }
  std::string get_description() const override { return std::string(PlayerT::kDescription); }
  Player* generate(core::game_slot_index_t) override { return new PlayerT(&pool_, params_); }
  void print_help(std::ostream& s) override { params_.make_options_description().print(s); }
  void parse_args(const std::vector<std::string>& args) override {
    namespace po2 = boost_util::program_options;
    po2::parse_args(params_.make_options_description(), args);

    size_t capacity = params_.num_procs;
    pool_.set_capacity(capacity);
  }

 private:
  PlayerParams params_ = PlayerT::default_params();
  UciPool& pool_;
};

}  // namespace a0achess

namespace core {

template <typename PlayerT>
class PlayerSubfactory<a0achess::UciPlayerGenerator<PlayerT>>
    : public PlayerSubfactoryBase<a0achess::Game> {
 public:
  using UciPool = typename a0achess::UciPlayerGenerator<PlayerT>::UciPool;

  a0achess::UciPlayerGenerator<PlayerT>* create(GameServerBase*) override {
    return new a0achess::UciPlayerGenerator<PlayerT>(pool_);
  }

 private:
  UciPool pool_;
};

}  // namespace core
