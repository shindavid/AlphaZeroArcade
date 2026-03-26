/*
 * Fathom tbconfig.h override.
 *
 * This file is picked up via -I include path priority over the default
 * tbconfig.h shipped with Fathom (which uses angle-bracket include).
 */

#ifndef TBCONFIG_H
#define TBCONFIG_H

#define TB_NO_HELPER_API

/*
 * Scoring constants.
 */
#define TB_VALUE_PAWN     100
#define TB_VALUE_MATE     32000
#define TB_VALUE_INFINITE 32767
#define TB_VALUE_DRAW     0
#define TB_MAX_MATE_PLY   255

#endif  /* TBCONFIG_H */
