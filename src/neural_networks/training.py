import copy

import numpy as np


class Fitter:

    def __init__(self, net, loss_criterion, optimiser):
        self.optimiser = optimiser
        self.loss_criterion = loss_criterion
        self.net = net
        self.best_state = None
        self.history = {"loss": [], "val_loss": []}

    def step(self, x, y, mode='train'):
        """

        :param net: pytorch module
        :param x: inputs
        :param y: outputs
        :param loss_criterion: loss function
        :param optimiser: optimiser
        :param mode: train or eva;
        :return:
        """

        self.net.train() if mode == 'train' else self.net.eval()

        self.net.zero_grad()
        out = self.net.forward(x)
        loss = self.loss_criterion(out, y)

        if mode == 'eval':
            return out, loss

        loss.backward()
        self.optimiser.step()

        return out, loss

    def reset_history(self):
        self.history = {"loss": [], "val_loss": []}


    def fit(self, data_generator, epochs=1):

        self.reset_history()

        epoch_loss = []
        for epoch in range(epochs):
            for batch_x, batch_y in data_generator:
                _, batch_loss = self.step(batch_x, batch_y,
                                                  mode="train")

                epoch_loss.append(batch_loss.item())

            self.history['loss'].append(np.average(epoch_loss))

        return self.history

    def advanced_fit(self, data_generator, validation_data=None,
                     epochs=1, early_stopping_patience=0):

        best_val_loss = np.inf
        epochs_without_improvement = 0
        self.reset_history()

        if not validation_data and early_stopping_patience:
            print('Early stopping is not possible without validation data')

        for epoch in range(epochs):

            epoch_losses = []
            for batch_x, batch_y in data_generator:
                batch_out, batch_loss = self.step(batch_x, batch_y,
                                                  mode="train")

                # Collect metrics
                epoch_losses.append(batch_loss.item())

            self.history['loss'].append(np.average(epoch_losses))

            if validation_data is not None:
                val_out, val_loss = self.step(validation_data[0],
                        validation_data[1], mode="eval")

                self.history['val_loss'].append(val_loss.item())

                if self.history['val_loss'][-1] < best_val_loss:
                    print(f"Saving model with val loss {self.history['val_loss'][-1]}")
                    self.best_state = copy.deepcopy(self.net.state_dict())
                    best_val_loss = self.history['val_loss'][-1]
                elif epochs_without_improvement + 1 > early_stopping_patience:
                    break
                else:
                    early_stopping_patience += 1

            metric_string = [f"{k}: {v[-1]}" for k, v in self.history.items() if
                             v != []]

            metric_string = ", ".join(metric_string)
            print(f"Epoch {epoch} - {metric_string}\n")

        return self.history, self.best_state
