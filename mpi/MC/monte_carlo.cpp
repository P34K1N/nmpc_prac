#include <mpi.h>

#include <fstream>
#include <iostream>
#include <random>
#include <utility>

class Walker {
 public:
  Walker(int left_border, int right_border, int start_pos, float right_prob)
      : left_border_(left_border),
        right_border_(right_border),
        start_pos_(start_pos),
        right_prob_(right_prob) {}

  std::pair<bool, int> walk() const {
    size_t step_count = 0;
    int pos = start_pos_;
    while (true) {
      pos += step();
      step_count += 1;
      if (pos == left_border_) {
        return std::make_pair(false, step_count);
      }
      if (pos == right_border_) {
        return std::make_pair(true, step_count);
      }
    }
  }

 private:
  int left_border_;
  int right_border_;
  int start_pos_;
  float right_prob_;

  int step() const {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::bernoulli_distribution walker(right_prob_);

    if (walker(rng)) {
      return 1;
    }
    return -1;
  }
};

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int myrank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (argc <= 5) {
    std::cerr << "Too few arguments" << std::endl;
    return 1;
  }

  int left_border = std::strtol(argv[1], NULL, 10);
  int right_border = std::strtol(argv[2], NULL, 10);
  int start_pos = std::strtol(argv[3], NULL, 10);
  float right_prob = std::strtof(argv[4], NULL);
  size_t exp_count = std::strtoul(argv[5], NULL, 10);

  MPI_Barrier(MPI_COMM_WORLD);
  double start, finish;
  start = MPI_Wtime();

  const Walker walker(left_border, right_border, start_pos, right_prob);

  int exp_count_for_process =
      exp_count / size + (exp_count % size > myrank ? 1 : 0);

  uint32_t total_right_count = 0;
  uint32_t total_steps = 0;
  for (size_t i = 0; i < exp_count_for_process; i++) {
    auto result = walker.walk();
    total_right_count += result.first ? 1 : 0;
    total_steps += result.second;
  }

  uint32_t reduced_total_right_count = 0;
  uint32_t reduced_total_steps = 0;

  MPI_Reduce(&total_right_count, &reduced_total_right_count, 1, MPI_UINT32_T,
             MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&total_steps, &reduced_total_steps, 1, MPI_UINT32_T, MPI_SUM, 0,
             MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  finish = MPI_Wtime();

  if (myrank == 0) {
    std::ofstream fout("output.txt");
    fout << float(reduced_total_right_count) / exp_count << " "
         << float(reduced_total_steps) / exp_count << std::endl;

    std::ofstream fstat("stat.txt");
    fstat << left_border << " " << right_border << " " << start_pos << " "
          << right_prob << " " << exp_count << " " << size << " "
          << finish - start << std::endl;
  }

  MPI_Finalize();
}
