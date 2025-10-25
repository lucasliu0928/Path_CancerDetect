#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:37:19 2025

@author: jliu6
"""

import matplotlib.pyplot as plt
import os


def plot_loss(train_loss, val_loss, save_path):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss", marker='o')
    plt.plot(range(1, len(train_loss) + 1), val_loss, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    



