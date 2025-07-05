#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>

namespace py = pybind11;

py::array_t<int> create_empty_board(int rows, int cols) {
    auto result = py::array_t<int>({rows, cols});
    auto buf = result.mutable_unchecked<2>();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            buf(i, j) = 0;
        }
    }

    return result;
}

py::array_t<int> copy_board(py::array_t<int> arr) {
    auto buf = arr.unchecked<2>();
    auto result = py::array_t<int>({buf.shape(0), buf.shape(1)});
    auto out = result.mutable_unchecked<2>();

    for (ssize_t i = 0; i < buf.shape(0); ++i) {
        for (ssize_t j = 0; j < buf.shape(1); ++j) {
            out(i, j) = buf(i, j);
        }
    }

    return result;
}

void apply_gravity_to_cols(py::array_t<int> arr, int row_last, int col_left, int col_right) {
    auto buf = arr.mutable_unchecked<2>();
    int num_rows = buf.shape(0);
    int num_cols = buf.shape(1);

    for (int r = row_last; r > 0; --r) {
        for (int c = col_left; c <= col_right; ++c) {
            if (buf(r, c) == 0 && buf(r - 1, c) != 0) {
                // Move the value down if the cell above is not empty
                buf(r, c) = buf(r - 1, c);
                buf(r - 1, c) = 0; // Clear the cell above
            }
        }
    }
}

void resolve_board_inplace(py::array_t<int> arr, int target_col, int max_value) {
    auto buf = arr.mutable_unchecked<2>();
    int num_rows = buf.shape(0);
    int num_cols = buf.shape(1);

    assert(num_cols == 5 && "Expected 5 columns in the grid");
    assert(0 <= target_col && target_col < num_cols && "Invalid target column");

    // Assume that gravity does not need to be resolved right away.
    bool any_changes = true;
    while (any_changes) {
        bool found_four_way_merge = false; int found_four_way_merge_row = -1; int found_four_way_merge_col = -1;
        bool found_three_way_l_merge = false; int found_three_way_l_merge_row = -1; int found_three_way_l_merge_col = -1;
        bool found_two_way_vertical_merge = false; int found_two_way_vertical_merge_row = -1; int found_two_way_vertical_merge_col = -1;
        bool found_three_way_horizontal_merge = false; int found_three_way_horizontal_merge_row = -1; int found_three_way_horizontal_merge_col = -1;
        bool found_two_way_horizontal_merge = false; int found_two_way_horizontal_merge_row = -1; int found_two_way_horizontal_merge_col = -1;

        // First pass: look for different kinds of merges and note their positions
        for (int r = 0; r < num_rows; ++r) {
            for (int c = 0; c < num_cols; ++c) {
                // For each cell, check if its neighbors can merge into it

                if (buf(r, c) == 0) continue; // Skip empty cells
                if (buf(r, c) >= max_value) continue; // Skip cells that are already at max value

                int center = buf(r, c);
                int left = (c > 0) ? buf(r, c - 1) : 0;
                int right = (c < num_cols - 1) ? buf(r, c + 1) : 0;
                int down = (r < num_rows - 1) ? buf(r + 1, c) : 0;

                if (!found_four_way_merge && left == center && right == center && down == center && center * 8 <= max_value) {
                    // Found a four-way merge (left, center, right, down)
                    found_four_way_merge = true;
                    found_four_way_merge_row = r;
                    found_four_way_merge_col = c;
                } else if (!found_three_way_l_merge && left == center && down == center && center * 4 <= max_value) {
                    // Found a three-way L merge (left, down, center)
                    found_three_way_l_merge = true;
                    found_three_way_l_merge_row = r;
                    found_three_way_l_merge_col = c;
                } else if (!found_three_way_l_merge && center == right && down == center && center * 4 <= max_value) {
                    // Found a three-way L merge (center, right, down)
                    found_three_way_l_merge = true;
                    found_three_way_l_merge_row = r;
                    found_three_way_l_merge_col = c;
                } else if (!found_two_way_vertical_merge && center == down && center * 2 <= max_value) {
                    // Found a two-way vertical merge (center, down)
                    found_two_way_vertical_merge = true;
                    found_two_way_vertical_merge_row = r;
                    found_two_way_vertical_merge_col = c;
                } else if (!found_three_way_horizontal_merge && left == center && right == center && center * 4 <= max_value) {
                    // Found a three-way horizontal merge (left, center, right)
                    found_three_way_horizontal_merge = true;
                    found_three_way_horizontal_merge_row = r;
                    found_three_way_horizontal_merge_col = c;
                } else if (!found_two_way_horizontal_merge && left == center && center * 2 <= max_value) {
                    // Found a two-way horizontal merge (left, center)
                    found_two_way_horizontal_merge = true;
                    found_two_way_horizontal_merge_row = r;
                    found_two_way_horizontal_merge_col = c;
                } else if (!found_two_way_horizontal_merge && center == right && center * 2 <= max_value) {
                    // Found a two-way horizontal merge (center, right)
                    found_two_way_horizontal_merge = true;
                    found_two_way_horizontal_merge_row = r;
                    found_two_way_horizontal_merge_col = c;
                }
            }
        }
    
        // Second pass: perform the merges based on the positions noted
        any_changes = found_four_way_merge || found_three_way_l_merge || found_two_way_vertical_merge || found_three_way_horizontal_merge || found_two_way_horizontal_merge;
        if (found_four_way_merge) {
            // Perform four-way merge
            int new_value = buf(found_four_way_merge_row, found_four_way_merge_col) * 8;
            buf(found_four_way_merge_row + 1, found_four_way_merge_col) = new_value; // Update down
            buf(found_four_way_merge_row, found_four_way_merge_col - 1) = 0; // Clear left
            buf(found_four_way_merge_row, found_four_way_merge_col + 1) = 0; // Clear right
            buf(found_four_way_merge_row, found_four_way_merge_col) = 0; // Clear center

            // Apply gravity to the left, center, and right columns
            apply_gravity_to_cols(arr, found_four_way_merge_row, found_four_way_merge_col - 1, found_four_way_merge_col + 1);
        } else if (found_three_way_l_merge) {
            // Perform three-way L merge
            int prev_value = buf(found_three_way_l_merge_row, found_three_way_l_merge_col);
            int new_value = prev_value * 4;
            buf(found_three_way_l_merge_row, found_three_way_l_merge_col) = 0; // Clear center
            buf(found_three_way_l_merge_row + 1, found_three_way_l_merge_col) = new_value; // Update down
            if (found_three_way_l_merge_col > 0 && buf(found_three_way_l_merge_row, found_three_way_l_merge_col - 1) == prev_value) {
                buf(found_three_way_l_merge_row, found_three_way_l_merge_col - 1) = 0; // Clear left
                apply_gravity_to_cols(arr, found_three_way_l_merge_row, found_three_way_l_merge_col - 1, found_three_way_l_merge_col);
            } else if (found_three_way_l_merge_col < num_cols - 1 && buf(found_three_way_l_merge_row, found_three_way_l_merge_col + 1) == prev_value) {
                buf(found_three_way_l_merge_row, found_three_way_l_merge_col + 1) = 0; // Clear right
                apply_gravity_to_cols(arr, found_three_way_l_merge_row, found_three_way_l_merge_col, found_three_way_l_merge_col + 1);
            }
        } else if (found_two_way_vertical_merge) {
            // Perform two-way vertical merge
            int new_value = buf(found_two_way_vertical_merge_row, found_two_way_vertical_merge_col) * 2;
            buf(found_two_way_vertical_merge_row + 1, found_two_way_vertical_merge_col) = new_value; // Update down
            buf(found_two_way_vertical_merge_row, found_two_way_vertical_merge_col) = 0; // Clear center
            apply_gravity_to_cols(arr, found_two_way_vertical_merge_row, found_two_way_vertical_merge_col, found_two_way_vertical_merge_col);
        } else if (found_three_way_horizontal_merge) {
            // Perform three-way horizontal merge
            int new_value = buf(found_three_way_horizontal_merge_row, found_three_way_horizontal_merge_col) * 4;
            buf(found_three_way_horizontal_merge_row, found_three_way_horizontal_merge_col - 1) = 0; // Clear left
            buf(found_three_way_horizontal_merge_row, found_three_way_horizontal_merge_col + 1) = 0; // Clear right
            buf(found_three_way_horizontal_merge_row, found_three_way_horizontal_merge_col) = new_value; // Update center
            apply_gravity_to_cols(arr, found_three_way_horizontal_merge_row, found_three_way_horizontal_merge_col - 1, found_three_way_horizontal_merge_col + 1);
        } else if (found_two_way_horizontal_merge) {
            // Perform two-way horizontal merge
            int prev_value = buf(found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col);
            int new_value = prev_value * 2;
            if (found_two_way_horizontal_merge_col > 0 && buf(found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col - 1) == prev_value) {
                if (found_two_way_horizontal_merge_col <= target_col) {
                    // If the merge is to the left of the target column, we merge 'left' into 'center'
                    buf(found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col - 1) = 0; // Clear left
                    buf(found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col) = new_value; // Update center
                } else {
                    // If the merge is to the right of the target column, we move 'center' into 'left'
                    buf(found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col) = 0; // Clear center
                    buf(found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col - 1) = new_value; // Update left
                }
                apply_gravity_to_cols(arr, found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col - 1, found_two_way_horizontal_merge_col);
            } else if (found_two_way_horizontal_merge_col < num_cols - 1 && buf(found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col + 1) == prev_value) {
                if (found_two_way_horizontal_merge_col >= target_col) {
                    // If the merge is to the right of the target column, we merge 'right' into 'center'
                    buf(found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col + 1) = 0; // Clear right
                    buf(found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col) = new_value; // Update center
                } else {
                    // If the merge is to the left of the target column, we move 'center' into 'right'
                    buf(found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col) = 0; // Clear center
                    buf(found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col + 1) = new_value; // Update right
                }
                apply_gravity_to_cols(arr, found_two_way_horizontal_merge_row, found_two_way_horizontal_merge_col, found_two_way_horizontal_merge_col + 1);
            }
        }
    }
}

bool step_inplace(py::array_t<int> arr, int target_col, int current_tile, int max_value) {
    int num_rows = arr.shape(0);
    int num_cols = arr.shape(1);
    auto buf = arr.mutable_unchecked<2>();

    // Try to add the tile from the top, in the target column
    if (buf(0, target_col) != 0) {
        // If the target column is already occupied:
        if (buf(0, target_col) == current_tile && buf(0, target_col) * 2 <= max_value) {
            // If the tile matches, we can merge
            buf(0, target_col) *= 2; // Merge the tile
        } else {
            return false; // Invalid move, cannot place here
        }
    } else {
        // If the target column is empty, place it at the first available row
        for (int r = num_rows - 1; r >= 0; --r) {
            if (buf(r, target_col) == 0) {
                buf(r, target_col) = current_tile; // Place the tile
                break;
            }
        }
    }

    // Resolve the board in-place based on the game rules
    resolve_board_inplace(arr, target_col, max_value);

    return true; // Move was successful
}

std::tuple<bool, py::array_t<int>> step_immutable(py::array_t<int> arr, int target_col, int current_tile, int max_value) {
    // Create a copy of the board
    auto new_board = copy_board(arr);
    bool result = step_inplace(new_board, target_col, current_tile, max_value);
    return {result, new_board};
}

float value_function(py::array_t<int> arr) {
    int num_rows = arr.shape(0);
    int num_cols = arr.shape(1);
    auto buf = arr.unchecked<2>();
    float score = 0.0f;

    for (int r = 0; r < num_rows; ++r) {
        for (int c = 0; c < num_cols; ++c) {
            int tile_value = buf(r, c);
            if (tile_value > 0) {
                // Penalize for each tile on the board
                score -= 1;

                // Add bonus for tiles that are above their successor in the merge chain
                if (r < num_rows - 1 && buf(r + 1, c) == tile_value * 2) {
                    score += 0.75f; // Bonus for being above a double tile
                }
            }
        }
    }

    // Punish non-decreasing columns
    for (int c = 0; c < num_cols; ++c) {
        for (int r = num_rows - 1; r > 0; --r) {
            if (buf(r, c) > 0 && buf(r, c) < buf(r - 1, c)) {
                score -= 10; // Penalty for non-decreasing column
                // break; // Only need to penalize once per column
            }
        }
    }

    return score;
}

// Mutate in-place: add 1 to every element
void increment_inplace(py::array_t<int> arr) {
    auto buf = arr.mutable_unchecked<2>();
    for (ssize_t i = 0; i < buf.shape(0); ++i)
        for (ssize_t j = 0; j < buf.shape(1); ++j)
            buf(i, j) += 1;
}

// Return a new array: double every element
py::array_t<int> double_array(py::array_t<int> arr) {
    auto buf = arr.unchecked<2>();
    auto result = py::array_t<int>({buf.shape(0), buf.shape(1)});
    auto out = result.mutable_unchecked<2>();
    for (ssize_t i = 0; i < buf.shape(0); ++i)
        for (ssize_t j = 0; j < buf.shape(1); ++j)
            out(i, j) = 2 * buf(i, j);
    return result;
}

// Return a scalar: sum of all elements
int sum_array(py::array_t<int> arr) {
    auto buf = arr.unchecked<2>();
    int total = 0;
    for (ssize_t i = 0; i < buf.shape(0); ++i)
        for (ssize_t j = 0; j < buf.shape(1); ++j)
            total += buf(i, j);
    return total;
}

// Example: wrap a numpy array in a C++ class
class GridWrapper {
  public:
    GridWrapper(py::array_t<int> arr) : arr_(arr) {}
    void add_scalar(int v) {
        auto buf = arr_.mutable_unchecked<2>();
        for (ssize_t i = 0; i < buf.shape(0); ++i)
            for (ssize_t j = 0; j < buf.shape(1); ++j)
                buf(i, j) += v;
    }
    py::array_t<int> get() const { return arr_; }

  private:
    py::array_t<int> arr_;
};

// Example: create a C++-owned array and expose as a class
class OwnedGrid {
  public:
    OwnedGrid(int rows, int cols) : data_(rows * cols, 0), rows_(rows), cols_(cols) {}
    void set(int i, int j, int v) { data_[i * cols_ + j] = v; }
    int get(int i, int j) const { return data_[i * cols_ + j]; }
    py::array_t<int> as_numpy() const {
        return py::array_t<int>({rows_, cols_}, data_.data());
    }
    py::array_t<int> as_numpy_mut() {
        // Create a capsule to keep 'this' alive as long as the numpy array exists
        return py::array_t<int>(
            {rows_, cols_},
            data_.data(),
            py::capsule(this, "OwnedGrid")
        );
    }

  private:
    std::vector<int> data_;
    int rows_, cols_;
};

PYBIND11_MODULE(cpp_core, m) {
    m.doc() = "C++ core for DropMergeRL";

    // EXAMPLE METHODS FOR REFERENCE
    m.def("increment_inplace", &increment_inplace, "Increment all elements in-place");
    m.def("double_array", &double_array, "Return a new array with all elements doubled");
    m.def("sum_array", &sum_array, "Return the sum of all elements");
    py::class_<GridWrapper>(m, "GridWrapper")
        .def(py::init<py::array_t<int>>())
        .def("add_scalar", &GridWrapper::add_scalar)
        .def("get", &GridWrapper::get);
    py::class_<OwnedGrid>(m, "OwnedGrid")
        .def(py::init<int, int>())
        .def("set", &OwnedGrid::set)
        .def("get", &OwnedGrid::get)
        .def("as_numpy", &OwnedGrid::as_numpy)
        .def("as_numpy_mut", &OwnedGrid::as_numpy_mut);

    // GAME LOGIC METHODS
    m.def("create_empty_board", &create_empty_board, "Create an empty board with specified rows and columns");
    m.def("copy_board", &copy_board, "Create a copy of the board");
    m.def("step_inplace", &step_inplace, "Perform a step in-place on the board");
    m.def("step_immutable", &step_immutable, "Perform a step on the board and return a new board");
    m.def("resolve_board_inplace", &resolve_board_inplace, "Resolve the board in-place based on the game rules");
    m.def("value_function", &value_function, "Calculate the value function for the board");
}
