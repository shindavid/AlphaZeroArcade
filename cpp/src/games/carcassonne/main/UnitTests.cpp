#include <games/carcassonne/BasicTypes.hpp>
#include <games/carcassonne/Constants.hpp>

#include <cstring>

/*
 * Tests carcassonne code.
 */

using namespace carcassonne;

int global_pass_count = 0;
int global_fail_count = 0;

void test_basic_types() {
  printf("Running %s()...\n", __func__);

  const char* expected_output[] = {"CCCC110 N 1 CCCC N",
                                   "CCCC110 E 0",
                                   "CCCC110 S 0",
                                   "CCCC110 W 0",
                                   "CCCF100 N 1 CCCF N|SW",
                                   "CCCF100 E 1 FCCC E|NW",
                                   "CCCF100 S 1 CFCC S|NE",
                                   "CCCF100 W 1 CCFC W|SE",
                                   "CCCF110 N 1 CCCF N|SW",
                                   "CCCF110 E 1 FCCC E|NW",
                                   "CCCF110 S 1 CFCC S|NE",
                                   "CCCF110 W 1 CCFC W|SE",
                                   "CCCR100 N 1 CCCR N|W|SW|NW",
                                   "CCCR100 E 1 RCCC N|E|NE|NW",
                                   "CCCR100 S 1 CRCC E|S|NE|SE",
                                   "CCCR100 W 1 CCRC S|W|SE|SW",
                                   "CCCR110 N 1 CCCR N|W|SW|NW",
                                   "CCCR110 E 1 RCCC N|E|NE|NW",
                                   "CCCR110 S 1 CRCC E|S|NE|SE",
                                   "CCCR110 W 1 CCRC S|W|SE|SW",
                                   "CCFF100 N 1 CCFF N|SE",
                                   "CCFF100 E 1 FCCF E|SW",
                                   "CCFF100 S 1 FFCC S|NW",
                                   "CCFF100 W 1 CFFC W|NE",
                                   "CCFF110 N 1 CCFF N|SE",
                                   "CCFF110 E 1 FCCF E|SW",
                                   "CCFF110 S 1 FFCC S|NW",
                                   "CCFF110 W 1 CFFC W|NE",
                                   "CCFF200 N 1 CCFF N|E|NE",
                                   "CCFF200 E 1 FCCF E|S|SE",
                                   "CCFF200 S 1 FFCC S|W|SW",
                                   "CCFF200 W 1 CFFC N|W|NW",
                                   "CCRR100 N 1 CCRR N|S|SE|SW",
                                   "CCRR100 E 1 RCCR E|W|SW|NW",
                                   "CCRR100 S 1 RRCC N|S|NE|NW",
                                   "CCRR100 W 1 CRRC E|W|NE|SE",
                                   "CCRR110 N 1 CCRR N|S|SE|SW",
                                   "CCRR110 E 1 RCCR E|W|SW|NW",
                                   "CCRR110 S 1 RRCC N|S|NE|NW",
                                   "CCRR110 W 1 CRRC E|W|NE|SE",
                                   "CFCF100 N 1 CFCF N|NE|SW",
                                   "CFCF100 E 1 FCFC E|SE|NW",
                                   "CFCF100 S 0",
                                   "CFCF100 W 0",
                                   "CFCF110 N 1 CFCF N|NE|SW",
                                   "CFCF110 E 1 FCFC E|SE|NW",
                                   "CFCF110 S 0",
                                   "CFCF110 W 0",
                                   "CFCF200 N 1 CFCF N|S|NE",
                                   "CFCF200 E 1 FCFC E|W|SE",
                                   "CFCF200 S 0",
                                   "CFCF200 W 0",
                                   "CFFF100 N 1 CFFF N|NE",
                                   "CFFF100 E 1 FCFF E|SE",
                                   "CFFF100 S 1 FFCF S|SW",
                                   "CFFF100 W 1 FFFC W|NW",
                                   "CFRR100 N 1 CFRR N|S|NE|SW",
                                   "CFRR100 E 1 RCFR E|W|SE|NW",
                                   "CFRR100 S 1 RRCF N|S|NE|SW",
                                   "CFRR100 W 1 FRRC E|W|SE|NW",
                                   "CRFR100 N 1 CRFR N|E|NE|SE",
                                   "CRFR100 E 1 RCRF E|S|SE|SW",
                                   "CRFR100 S 1 FRCR S|W|SW|NW",
                                   "CRFR100 W 1 RFRC N|W|NE|NW",
                                   "CRRF100 N 1 CRRF N|E|NE|SE",
                                   "CRRF100 E 1 FCRR E|S|SE|SW",
                                   "CRRF100 S 1 RFCR S|W|SW|NW",
                                   "CRRF100 W 1 RRFC N|W|NE|NW",
                                   "CRRR100 N 1 CRRR N|E|S|W|NE|SE|SW",
                                   "CRRR100 E 1 RCRR N|E|S|W|SE|SW|NW",
                                   "CRRR100 S 1 RRCR N|E|S|W|NE|SW|NW",
                                   "CRRR100 W 1 RRRC N|E|S|W|NE|SE|NW",
                                   "FFFF001 N 1 FFFF NE|X",
                                   "FFFF001 E 0",
                                   "FFFF001 S 0",
                                   "FFFF001 W 0",
                                   "FFFR001 N 1 FFFR W|NE|X",
                                   "FFFR001 E 1 RFFF N|SE|X",
                                   "FFFR001 S 1 FRFF E|SW|X",
                                   "FFFR001 W 1 FFRF S|NW|X",
                                   "FFRR000 N 1 FFRR S|NE|SW",
                                   "FFRR000 E 1 RFFR W|SE|NW",
                                   "FFRR000 S 1 RRFF N|NE|SW",
                                   "FFRR000 W 1 FRRF E|SE|NW",
                                   "FRFR000 N 1 FRFR E|NE|SE",
                                   "FRFR000 E 1 RFRF S|SE|SW",
                                   "FRFR000 S 0",
                                   "FRFR000 W 0",
                                   "FRRR000 N 1 FRRR E|S|W|NE|SE|SW",
                                   "FRRR000 E 1 RFRR N|S|W|SE|SW|NW",
                                   "FRRR000 S 1 RRFR N|E|W|NE|SW|NW",
                                   "FRRR000 W 1 RRRF N|E|S|NE|SE|NW",
                                   "RRRR000 N 1 RRRR N|E|S|W|NE|SE|SW|NW",
                                   "RRRR000 E 0",
                                   "RRRR000 S 0",
                                   "RRRR000 W 0"};

  int r = 0;
  bool fail = false;
  for (int t = 0; t < kNumTileTypes; ++t) {
    TileType tile = TileType(t);
    for (int d = 0; d < 4; ++d) {
      const char* expected = expected_output[r++];
      Direction dir = Direction(d);
      bool valid = is_valid_orientation(tile, dir);

      EdgeProfile edge_profile = get_edge_profile(tile, dir);
      MeepleLocationProfile meeple_profile = get_valid_meeple_placements(tile, dir);

      char buffer[1024];
      if (!valid) {
        sprintf(buffer, "%s %s %d", tile_type_to_string(tile), direction_to_string(dir), valid);
      } else {
        sprintf(buffer, "%s %s %d %s %s", tile_type_to_string(tile), direction_to_string(dir),
                valid, edge_profile.to_string().c_str(), meeple_profile.to_string().c_str());
      }

      if (std::strcmp(buffer, expected) != 0) {
        printf("\n");
        printf("Failure!");
        printf("Expected: %s\n", expected);
        printf("  Actual: %s\n", buffer);
        fail = true;
      }
    }
  }
  if (fail) {
    global_fail_count++;
  } else {
    global_pass_count++;
  }
}

int main() {
  test_basic_types();

  if (global_fail_count > 0) {
    int total_count = global_pass_count + global_fail_count;
    printf("Failed %d of %d test%s!\n", global_fail_count, total_count, total_count > 1 ? "s" : "");
  } else {
    printf("All tests passed (%d of %d)!\n", global_pass_count, global_pass_count);
  }
  return 0;
}
