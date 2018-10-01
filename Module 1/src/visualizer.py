import matplotlib.pyplot as plt


class Visualizer:

    @staticmethod
    def plot_error(error_history, validation_history):
        fig = plt.figure()
        fig.suptitle('Error and Validation', fontsize=18)
        # Change to proper format
        e_xs = [e[0] for e in error_history]
        e_ys = [e[1] for e in error_history]
        v_xs = [e[0] for e in validation_history]
        v_ys = [e[1] for e in validation_history]
        plt.plot(e_xs, e_ys, label='Error history')
        plt.plot(v_xs, v_ys, label='Validation error history')
        plt.xlabel("Steps")
        plt.ylabel("Error")
        plt.legend()
        plt.show()
