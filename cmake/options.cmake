# --------------------------------------------------------------------------- #
## Exectuble Options
# --------------------------------------------------------------------------- #

option(ENABLE_BUILD_CLI      "build the clvotess command line interface"  ON )
option(ENABLE_BUILD_PYVOTESS "build the pyvotess so library"              ON )
option(ENABLE_BUILD_TEST     "build the test executable"                  ON )
option(ENABLE_BUILD_REGR     "build the regression test executable"       ON )

# --------------------------------------------------------------------------- #
## Build Options
# --------------------------------------------------------------------------- #

option(ENABLE_DEBUG          "build project in debug mode"                OFF )

# --------------------------------------------------------------------------- #
## Compiler Options
# --------------------------------------------------------------------------- #

# NOTE: This option is deprecated. Supply compiler through command line instead
# option(USE_ACPP            "Uses acpp compiler instead of icpx default" OFF )

# --------------------------------------------------------------------------- #
## End
# --------------------------------------------------------------------------- #
