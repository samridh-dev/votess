#include <catch2/catch_session.hpp>
#include <iostream>
#include <string>
int main(int argc, char *args[]) {
  return Catch::Session().run(argc, args);
}
