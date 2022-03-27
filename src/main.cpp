#include <filesystem>
#include <memory>

#include "arg_parser.hpp"
#include "gradient_descent/parallel_sgd.hpp"
#include "gradient_descent/sgd.hpp"
#include "io.hpp"

using namespace LinearRegression;
namespace fs = std::filesystem;

void make_prediction(const po::variables_map& args)
{
    // Regression
    auto num_epochs = args["num-epochs"].as<unsigned>();
    auto learning_rate = args["lr"].as<double>();
    auto weight_decay = args["weight-decay"].as<double>();
    auto normalize = args["normalize"].as<bool>();

    std::unique_ptr<Interface> regression;
    if (!args["parallel"].as<bool>()) {
        regression = std::make_unique<SGD>(num_epochs, learning_rate, weight_decay, normalize);
    } else {
        auto num_threads = args["num-threads"].as<unsigned>();
        auto num_step_epochs = args["num-step-epochs"].as<unsigned>();
        regression = std::make_unique<ParallelSGD>(num_epochs, learning_rate, weight_decay, normalize, num_step_epochs, num_threads);
    }
    // Fit
    MatrixXd input = load_csv(args["input-path"].as<std::string>());
    VectorXd target = load_csv(args["target-path"].as<std::string>());
    regression->fit(input, target);

    // Predict
    MatrixXd eval = load_csv(args["eval-path"].as<std::string>());
    VectorXd prediction = regression->predict(eval);

    fs::path out_path(args["out-path"].as<std::string>());
    auto parent_dir = out_path.parent_path();

    if (!fs::is_directory(parent_dir) || !fs::exists(parent_dir)) {
        fs::create_directories(parent_dir);
    }
    dump_csv(out_path, prediction);
}

int main(int argc, char* argv[])
{
    make_prediction(parse_args(argc, argv));
    return 0;
}
