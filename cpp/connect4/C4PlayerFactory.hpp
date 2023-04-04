#pragma once

#include <map>
#include <string>
#include <utility>

#include <boost/program_options.hpp>

#include <common/AbstractPlayer.hpp>
#include <common/GameStateConcept.hpp>
#include <common/HumanTuiPlayer.hpp>
#include <common/Mcts.hpp>
#include <common/MctsPlayer.hpp>
#include <common/RandomPlayer.hpp>
#include <connect4/C4GameState.hpp>
#include <connect4/C4Tensorizor.hpp>
#include <util/CppUtil.hpp>
#include <util/MetaProgramming.hpp>

namespace common {

template <class Widget, class GameState>
concept PlayerWidgetConcept = requires(Widget widget) {
  { util::decay_copy(Widget::kTypeStrs[0]) } -> std::same_as<const char*>;
  { util::decay_copy(Widget::kDescription) } -> std::same_as<const char*>;
  { widget.print_help() };
  { widget.make_player_generator(std::vector<std::string>{}) } -> std::same_as<std::function<AbstractPlayer<GameState>*()>>;
};

template <GameStateConcept GameState>
struct TUIPlayerWidget {
  static constexpr const char* kTypeStrs[] = {"TUI"};
  static constexpr const char* kDescription = "human player";
  void print_help() {}
  std::function<AbstractPlayer<GameState>*()> make_player_generator(const std::vector<std::string>& args) {
    return []() { return new common::HumanTuiPlayer<GameState>(); };
  }
};

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor, bool Competitive>
struct MctsCompetitivePlayerWidgetBase {
  using Mcts = common::Mcts<GameState, Tensorizor>;
  using MctsPlayer = common::MctsPlayer<GameState, Tensorizor>;
  using MctsParams = typename Mcts::Params;
  using MctsPlayerParams = typename MctsPlayer::Params;

  MctsCompetitivePlayerWidgetBase()
  : mcts_params_(Competitive ? Mcts::kCompetitive : Mcts::kTraining)
  , mcts_player_params_(Competitive ? MctsPlayer::kCompetitive : MctsPlayer::kTraining)
  {}

  void print_help() {
    make_options_description().print(std::cout);
  }

  std::function<AbstractPlayer<GameState>*()> make_player_generator(const std::vector<std::string>& args) {
    namespace po = boost::program_options;

    auto desc = make_options_description();

    po::variables_map vm;
    po::store(po::command_line_parser(args).options(desc).run(), vm);
    po::notify(vm);

    return [&]() { return new MctsPlayer(mcts_player_params_, mcts_params_); };
  }

private:
  auto make_options_description() {
    return mcts_params_.make_options_description().add(mcts_player_params_.make_options_description());
  }
  MctsParams mcts_params_;
  MctsPlayerParams mcts_player_params_;
};

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
struct MctsCompetitivePlayerWidget : public MctsCompetitivePlayerWidgetBase<GameState, Tensorizor, true> {
  static constexpr const char* kTypeStrs[] = {"MCTS-C", "MCTS-Competitive"};
  static constexpr const char* kDescription = "MCTS player for competitive play";
};

template<GameStateConcept GameState, TensorizorConcept<GameState> Tensorizor>
struct MctsTrainingPlayerWidget : public MctsCompetitivePlayerWidgetBase<GameState, Tensorizor, false> {
  static constexpr const char* kTypeStrs[] = {"MCTS-T", "MCTS-Training"};
  static constexpr const char* kDescription = "MCTS player for training play";
};

template <GameStateConcept GameState>
struct RandomPlayerWidget {
  static constexpr const char* kTypeStrs[] = {"Random"};
  static constexpr const char* kDescription = "random player";
  void print_help() {}
  std::function<AbstractPlayer<GameState>*()> make_player_generator(const std::vector<std::string>&) {
    return []() { return new common::RandomPlayer<GameState>(); };
  }
};

template<GameStateConcept GameState_>
class PlayerFactoryBase {
public:
  using GameState = GameState_;
  using Player = AbstractPlayer<GameState>;
  using player_generator_t = std::function<Player*()>;

  struct player_generator_seat_t {
    player_generator_t generator;
    int seat;
  };
  using player_generator_seat_vec_t = std::vector<player_generator_seat_t>;

  struct Params {
    auto make_options_description();

    std::string type;
    int seat = -1;
  };
};

template<GameStateConcept GameState_, PlayerWidgetConcept<GameState_>... Widgets>
class PlayerFactory : public PlayerFactoryBase<GameState_> {
public:
  using WidgetTypes = mp::TypeList<Widgets...>;
  static constexpr int kNumWidgets = sizeof...(Widgets);
  static constexpr int kNumPlayers = GameState_::kNumPlayers;
  using WidgetTuple = std::tuple<Widgets...>;

  using base_t = PlayerFactoryBase<GameState_>;
  using GameState = base_t::GameState;
  using Player = base_t::Player;
  using player_generator_seat_t = base_t::player_generator_seat_t;
  using player_generator_seat_vec_t = base_t::player_generator_seat_vec_t;

  static player_generator_seat_vec_t parse(const std::vector<std::string>& player_strs);

  static void print_help(const std::vector<std::string>& player_strs);

private:
  template<PlayerWidgetConcept<GameState> Widget> static void matches_type(const std::string& type, bool& matches)
  {
    for (auto t : Widget::kTypeStrs) {
      if (type == t) {
        matches = true;
        return;
      }
    }
  }

  template<PlayerWidgetConcept<GameState> Widget> static void print_description() {
    std::cout << "  ";
    for (int k = 0; k < (int)sizeof(Widget::kTypeStrs) / 8; ++k) {
      if (k > 0) {
        std::cout << "/";
      }
      std::cout << Widget::kTypeStrs[k];
    }
    std::cout << ": " << Widget::kDescription << std::endl;
  }

  template<PlayerWidgetConcept<GameState> Widget> void print_type_options(bool print) {
    if (!print) return;

    std::cout << std::endl << "--type=";
    for (int k = 0; k < (int)sizeof(Widget::kTypeStrs) / 8; ++k) {
      if (k > 0) {
        std::cout << "/";
      }
      std::cout << Widget::kTypeStrs[k];
    }
    std::cout << " options:" << std::endl;

    constexpr int W = mp::IndexOf_v<WidgetTypes, Widget>;
    Widget& widget = std::get<W>(widgets_);
    widget.print_help();
  }

  template<PlayerWidgetConcept<GameState> Widget> void set_generator(
      const std::string& type, const std::vector<std::string>& tokens,
      player_generator_seat_t& player_generator_seat, bool& matched)
{
    if (matched) return;
    matches_type<Widget>(type, matched);
    if (!matched) return;
    constexpr int W = mp::IndexOf_v<WidgetTypes, Widget>;
    Widget& widget = std::get<W>(widgets_);
    player_generator_seat.generator = widget.make_player_generator(tokens);
  }

  static PlayerFactory* instance();

  player_generator_seat_t parse_helper(const std::string& player_str);

  static PlayerFactory* instance_;
  WidgetTuple widgets_;
};

}  // namespace common

namespace c4 {

using PlayerFactory = common::PlayerFactory<
    GameState,
    common::TUIPlayerWidget<GameState>,
    common::MctsCompetitivePlayerWidget<GameState, Tensorizor>,
    common::MctsTrainingPlayerWidget<GameState, Tensorizor>,
    common::RandomPlayerWidget<GameState>
    >;

}  // namespace c4

#include <connect4/inl/C4PlayerFactory.inl>
