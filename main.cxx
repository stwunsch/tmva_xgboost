#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>

using json = nlohmann::json;

json read_file(const std::string &filename) {
  std::ifstream i(filename);
  json j;
  i >> j;
  return j;
}

int main() {
  auto config = read_file("model.json");

  for (auto &tree : config)
    std::cout << tree << "\n\n";
}
