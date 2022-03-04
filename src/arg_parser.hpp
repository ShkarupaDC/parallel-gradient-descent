#pragma once

#include <boost/program_options.hpp>
namespace po = boost::program_options;

po::variables_map parse_args(int argc, char* argv[]);
