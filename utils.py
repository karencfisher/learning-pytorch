import numpy as np
import matplotlib.pyplot as plt
import random


class Normalize:
    def fit(self, x):
        self.avg = np.mean(x)
        self.std = np.std(x)

    def transform(self, x):
        return (x - self.avg) / self.std
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    
class GasGuage:
    def __init__(self, n_steps, percentage=5):
        self.n_steps = n_steps
        self.percentage = percentage
        self.size = 100 // percentage
        
    def begin(self):
        print(f'0/{self.n_steps}[{" " * self.size}]', end='')

    def update(self, step):
        percent = int(step / self.n_steps * 100)
        if percent % self.percentage == 0:
            done = percent // self.percentage
            left = self.size - done
            print(f'\r{step}/{self.n_steps}[{"=" * done}{" " * left}]', end='')

    def done(self, text):
        print(f' {text}')   

    
def time_str(seconds):
        hr = int(seconds / 3600)
        seconds %= 3600
        min = int(seconds / 60)
        sec = seconds % 60
        if hr > 0:
            output = f'{hr} hr {min} min {sec:.2f} sec'
        elif min > 0:
            output = f'{min} min {sec:.2f} sec'
        else:
            output = f'{sec:.4f} sec'
        return output

def show_samples(x, y, labels=None):
    imgs = []
    for _ in range(4):
        idx = random.randint(0, x.shape[0] - 1)
        imgs.append((x[idx], y[idx]))
    
    plt.figure(figsize=(5, 5))
    for idx, item in enumerate(imgs):
        image, label = item
        if labels is not None:
            label = labels[label]
        plt.subplot(2, 2, idx + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Label : {label}")
    plt.show()
    
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(history['loss'], color='red')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    
    accuracy = history.get('accuracy')
    if accuracy is not None:
        axes[1].plot(history['accuracy'], color='green')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')

    plt.show()


