import torch
import time
import utils


class ModelHarness:
    def __init__(self, model, loss_fn, optimizer, learning_rate=0.001):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer(model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f'Device = {self.device}')
        model.to(self.device) 

    def train(self, data, num_epochs):
        '''
        Train a pytorch model

        Inputs: data - torch dataloader
                num_epochs - number of epochs

        Returns: dictionary with average loss and accuracy for each epoch
        '''
        loss_hist = [0] * num_epochs
        accuracy_hist = [0] * num_epochs
        gg = utils.GasGuage(len(data))
        for epoch in range(num_epochs):
            self.model = self.model.train()
            print(f'Epoch {epoch + 1}')
            gg.begin()
            start = time.time()
            for count, batch in enumerate(data):
                # load batches to device (gpu or cpu)
                x_batch, y_batch = batch
                x_batch.to(self.device)
                y_batch.to(self.device)

                # run forward pass
                self.optimizer.zero_grad()
                pred = self.model(x_batch)

                # Get loss and run backward pass
                loss = self.loss_fn(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                
                # get stats
                loss_hist[epoch] += loss.item()
                if pred.shape[1] == 1:
                    accurate_preds = ((pred >= 0.5) == y_batch).float()
                else:
                    accurate_preds = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist[epoch] += accurate_preds.mean()

                # update gas guage
                gg.update(count + 1)
            
            # get mean stats
            loss_hist[epoch] /= len(data)
            accuracy_hist[epoch] /= len(data)

            # update guage
            elapsed = time.time() - start
            time_str = utils.time_str(elapsed)
            gg.done(f'{time_str} - Loss: {loss_hist[epoch]:.4f} - Accuracy: {accuracy_hist[epoch] * 100:.2f}%')
        return {'loss': loss_hist, 'accuracy': accuracy_hist}
    
    def evaluate(self, data):
        '''
        Infer on the model with batches of data

        Input: data - torch data loader

        Returns: logits/probs and labels
        '''
        preds, labels, accurates = [], [], []
        scores = 0
        gg = utils.GasGuage(len(data))
        gg.begin()
        start = time.time()
        self.model = self.model.eval()
        for count, batch in enumerate(data):
            
            # prepare data for device (gpu or cpu)
            x_batch, y_batch = batch
            x_batch.to(self.device)
            y_batch.to(self.device)

            # forward pass through model
            with torch.no_grad():
                preds_batch = self.model(x_batch)

            # accumulate preds
            preds.append(preds_batch.cpu())

            # get and accumulate labels
            if preds_batch.shape[1] == 1:
                labels_batch = ((preds_batch >= 0.5) == y_batch)
            else:
                labels_batch = torch.argmax(preds_batch, dim=1)
            labels.append(labels_batch)

            # And accurate labels
            accurate_batch = (labels_batch == y_batch).float()
            accurates.append(accurate_batch)

            # update guage
            gg.update(count + 1)

        # concatenate preds and labels
        preds_out = torch.cat(preds)
        labels_out = torch.cat(labels)
        score = torch.cat(accurates).mean()
        
        # update guage
        elapsed = time.time() - start
        gg.done(f'{utils.time_str(elapsed)} - Accuracy {score * 100:.2f}%')

        return preds_out, labels_out, score
    
    def infer(self, x):
        '''
        Inference on single batch (or example)
        if labels provided, also display and return accuracy
        '''
        self.model = self.model.eval()
        with torch.no_grad():
            preds = self.model(x)
        if preds.shape[1] == 1:
            labels = (preds >= 0.5)
        else:
            labels = torch.argmax(preds, dim=1)
        return preds.cpu(), labels
