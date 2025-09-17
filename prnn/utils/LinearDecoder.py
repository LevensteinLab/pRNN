#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:33:40 2022

@author: dl2820
"""

import torch
from torch import nn
import numpy as np


class linearDecoder:
    def __init__(self, numUnits, numX):
        """

        Parameters
        ----------
        numUnits : Number of units
        numX : Number of spatial locations

        """
        self.model = linnet(numUnits, numX)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=3e-1
        )
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fun = NLLLoss()   #Because softmax is applied in linnet.
        # If you change this remove log in trainstep

    def decode(self, h, withSoftmax=True):
        """
        Parameters
        ----------
        h : [Nt x Nunits] pytorch tensor
        withSoftmax : T/F, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        decodedX : TYPE
            The decoded location.
        p_X : TYPE
            Probabilty of being at each location.

        """
        p_X = self.model(h)
        decodedX = p_X.argmax(dim=1)
        if withSoftmax:
            sm = nn.Softmax(dim=1)
            p_X = sm(p_X)
        return decodedX, p_X

    def train(self, h, pos, batchSize=0.75, numBatches=10000, forceDevice=False):
        """
        Train the decoder from activity-position pairs

        Parameters
        ----------
        h : [Nt x Nunits] tensor
        pos : [Nt x numX] tensor
            The (binned/linearized) spatial position at each timestep
        batchSize : optional
            Fraction of data to use each learning step. Default: 0.5.
        numBatches : optional
            How many training steps. Default: 10000

        """
        # Consider: while loss doesn't change or is big enough...
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if forceDevice:
            device = forceDevice
        self.model.to(device)

        print(f"Training Decoder on {device}...")
        for step in range(numBatches):
            batch = np.random.choice(
                pos.shape[0], int(batchSize * pos.shape[0]), replace=False
            )
            h_batch, pos_batch = h[batch, :], pos[batch]
            h_batch, pos_batch = h_batch.to(device), pos_batch.to(device)
            steploss = self.trainstep(h_batch, pos_batch)
            if (100 * step / numBatches) % 10 == 0 or step == numBatches - 1:
                print(f"loss: {steploss:>f} [{step:>5d}\n{numBatches:>5d}]")

        print("Training Complete. Back to the cpu")
        self.model.to("cpu")
        return

    def trainstep(self, h_train, pos_train):
        """
        One training step of the decoder
        """
        decodedX, p_X = self.decode(h_train, withSoftmax=False)

        loss = self.loss_fn(p_X, pos_train)

        self.optimizer.zero_grad()  # Reset the gradients
        loss.backward()  # Backprop the gradients w.r.t loss
        self.optimizer.step()  # Update parameters one step

        steploss = loss.item()
        return steploss


class linnet(nn.Module):
    def __init__(self, numUnits, numX):
        super(linnet, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(numUnits, numX, bias=False),
        )

    def forward(self, x):
        logits = self.lin(x)
        return logits
