#include <fstream>
#include <iostream>
#include <string>

#include "gradient_descent/arg_parser.hpp"

po::variables_map parse_args(int argc, char* argv[])
{
    try {
        po::options_description cli_only("CLI options");
        // clang-format off
        cli_only.add_options()
            ("help", "produce help message")
            ("config", po::value<std::string>(), "path to config file");
        // clang-format on

        po::options_description algo("Algorithm options");
        // clang-format off
        algo.add_options()
            ("input-path", po::value<std::string>(), "path to NPY file with train and test data")
            ("num-epochs", po::value<int>()->default_value(1e3), "number of training epochs")
            ("lr", po::value<double>()->default_value(1e-3), "learning rate")
            ("weight-decay", po::value<double>()->default_value(1e-2), "L2 regularization lambda term")
            ("normalize", po::bool_switch(), "wether to normalize X");
        // clang-format on

        po::options_description general("Linear regression config");
        general.add(cli_only).add(algo);

        po::variables_map args;
        po::store(po::parse_command_line(argc, argv, general), args);
        po::notify(args);

        if (args.count("config_file")) {
            auto config_file = args["config_file"].as<std::string>();

            std::ifstream file(config_file.c_str());
            if (!file) {
                std::cerr << "Can not open config file " << config_file << std::endl;
            } else {
                po::store(po::parse_config_file(file, algo), args);
                po::notify(args);
            }
        }
        if (args.count("help")) {
            std::cout << general << std::endl;
            exit(0);
        }
        return args;
    } catch (std::exception& error) {
        std::cerr << error.what() << std::endl;
        exit(1);
    }
}
