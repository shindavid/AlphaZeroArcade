#include <core/GameLog.hpp>
#include <games/connect4/Game.hpp>
#include <games/connect4/Tensorizor.hpp>

using GameLogReader = core::GameLogReader<c4::Game>;

extern "C" {

core::ShapeInfo* get_shape_info_array() { return GameLogReader::get_shape_info(); }

void free_shape_info_array(core::ShapeInfo* info) { delete[] info; }

GameLogReader* GameLogReader_new(const char* filename) { return new GameLogReader(filename); }

void GameLogReader_delete(GameLogReader* log) { delete log; }

void GameLogReader_load(GameLogReader* log, int index, bool apply_symmetry, const char** keys,
                        float** values, int num_keys) {
  log->load(index, apply_symmetry, keys, values, num_keys);
}

}  // extern "C"
