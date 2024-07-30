# --------------------------------------------------------------------------- #
## Exectuble Options
# --------------------------------------------------------------------------- #

option(ENABLE_BUILD_CLI      "build the clvotess command line interface"  ON )
option(ENABLE_BUILD_PYVOTESS "build the pyvotess so library"              ON )
option(ENABLE_BUILD_TEST     "build the test executable"                  ON )

# --------------------------------------------------------------------------- #
## Build Options
# --------------------------------------------------------------------------- #

option(ENABLE_DEBUG          "build project in debug mode"                ON )

# --------------------------------------------------------------------------- #
## Compiler Options
# --------------------------------------------------------------------------- #

option(USE_ACPP              "Uses acpp compiler instead of icpx default" OFF )

# --------------------------------------------------------------------------- #
## End
# --------------------------------------------------------------------------- #
