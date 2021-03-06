#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

po::variables_map parse_args(int argc, char* argv[])
{
    try {
        po::options_description cli_only("CLI options");
        // clang-format off
        cli_only.add_options()
            ("help,h", "produce help message")
            ("config,c", po::value<std::string>(), "path to config file");
        // clang-format on

        po::options_description algo("Algorithm options");
        // clang-format off
        algo.add_options()
            ("input-path,i", po::value<std::string>()->required(), "path to input CSV file")
            ("target-path,t", po::value<std::string>()->required(), "path to target CSV file")
            ("eval-path,e", po::value<std::string>(), "path to evaluation CSV file")
            ("out-path,o", po::value<std::string>()->default_value("output.csv"), "path to output CSV file")
            ("cost-path", po::value<std::string>(), "path to cost CSV file")
            ("parallel,p", po::bool_switch(), "wether to use parallel or serial SGD")
            ("num-epochs,n", po::value<unsigned>()->default_value(1e3), "number of training epochs")
            ("batch-size,b", po::value<int>()->default_value(-1), "batch size for SGD")
            ("lr,l", po::value<double>()->default_value(1e-3), "learning rate")
            ("weight-decay,w", po::value<double>()->default_value(1e-2), "L2 regularization lambda term")
            ("normalize", po::bool_switch(), "wether to normalize input")
            ("num-threads", po::value<unsigned>()->default_value(std::max(1u, std::thread::hardware_concurrency() - 1)), "number of threads to use for parallel SGD")
            ("num-step-epochs", po::value<unsigned>()->default_value(1), "number of epochs to compute in each thread before weight sharing");
        // clang-format on

        po::options_description general("Linear regression config");
        general.add(cli_only).add(algo);

        po::variables_map args;
        po::store(po::parse_command_line(argc, argv, general), args);

        if (args.count("help")) {
            std::cout << general << std::endl;
            exit(0);
        }
        if (args.count("config")) {
            auto config_file = args["config"].as<std::string>();

            std::ifstream file(config_file.c_str());
            if (!file) {
                std::cerr << "Can not open config file " << config_file << std::endl;
            } else {
                po::store(po::parse_config_file(file, algo), args);
            }
        }
        po::notify(args);
        return args;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        exit(1);
    }
}
