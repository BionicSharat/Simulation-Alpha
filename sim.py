import random
import numpy as np
from model import *
import torch

# Plan
# 1. init every time
# 2. reward
# 3. play action
# 4. game iter

# init

MovingSpeed = 142.45 # pixels every turn

class SimulationAI:

    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.reset()
        self.Turn
        self.Attacks
        self.Map
        self.iceBergs

    def reset(self):
        self.Attacks = [] # [owner_start, end_loc, amountOfTroops, turns_till_arrive]
        self.Turn = 0

        self.Map = np.zeros((self.h, self.w))
        self.iceBergs = [
                {"id": 0, "Owner": 0, "l": 1, "loc": [1300,600], "troops": 11},
                {"id": 1, "Owner": -1, "l": 1, "loc": [1880,1800], "troops": 20},
                {"id": 2, "Owner": -1, "l": 1, "loc": [2433,1200], "troops": 10},
                {"id": 3, "Owner": -1, "l": 1, "loc": [2433,2600], "troops": 10},
                {"id": 4, "Owner": -1, "l": 1, "loc": [3967,1200], "troops": 10},
                {"id": 5, "Owner": -1, "l": 1, "loc": [3967,2600], "troops": 10},
                {"id": 6, "Owner": -1, "l": 1, "loc": [4520,1800], "troops": 20},
                {"id": 7, "Owner": 1, "l": 1, "loc": [1300,600], "troops": 11}
            ]

        for iceberg in self.iceBergs: 
            self.Map[iceberg["loc"][1] - 1][iceberg["loc"][0] - 1] = 1
    # GAME build func

    def checkWin(self):
        reward = 0
        if (len([1 for i in self.iceBergs if i["Owner"] == 0 or i["Owner"] == 1]) == 8) or self.iceBergs[0]["Owner"] == self.iceBergs[-1]["Owner"]:
            if self.iceBergs[-1]["Owner"] == 0:
                reward = 10
            elif self.iceBergs[0]["Owner"] == 1:
                reward = -10
            return True, reward
        else:
            return False, 0

    def indexById(self, idValue):
        for index, i in enumerate(self.iceBergs):
            if i["id"] == idValue:
                return index

    def turnsTillArrival(self, LocationStart, EndLocation):
        distance = ((LocationStart[0] - EndLocation[1])**2 + (LocationStart[1] - EndLocation[1])**2)**(1/2)
        return round(distance/MovingSpeed)

    def upgradeLevel(self, idValue):
        if (self.iceBergs[self.indexById(idValue)]["troops"] >= (self.iceBergs[self.indexById(idValue)]["l"] + 1) * 10):
            self.iceBergs[self.indexById(idValue)]["l"] += 1
            self.iceBergs[self.indexById(idValue)]["troops"] -= self.iceBergs[self.indexById(idValue)]["l"] * 10


    def sendTroops(self, idStart, idEnd, amountOfTroops):
        turns_till_arrive = self.turnsTillArrival(self.iceBergs[self.indexById(idStart)]["loc"], self.iceBergs[self.indexById(idEnd)]["loc"])
        end_loc = self.iceBergs[self.indexById(idEnd)]["loc"]
        owner_start = self.iceBergs[idStart]["Owner"]
        if (owner_start == 0):
            self.Attacks.append([owner_start, end_loc, amountOfTroops, turns_till_arrive])

    def checkTroopsReached(self):
        Arrived_l = []
        for indexAttack, group in enumerate(self.Attacks):
            if group[-1] == 1:
                for index, i in enumerate(self.iceBergs):
                    if i["loc"] == group[1]:
                        if self.iceBergs[index]["troops"] - group[2] < 0:
                            self.iceBergs[index]["Owner"] = group[0]
                            self.iceBergs[index]["troops"] = abs(self.iceBergs[index]["troops"] - group[2])
                            Arrived_l.append(indexAttack)
                        else: 
                            self.iceBergs[index]["troops"] =- group[2]
            else: 
                group[-1] -= 1
        deleted_l = self.Attacks
        for index in sorted(Arrived_l, reverse=True):
            del deleted_l[index]
        self.Attacks = deleted_l

    def play(self, actions = []):
        try:
            for i in range(0, 21, 3):
                self.sendTroops(actions[i], actions[i+1], actions[i+2])
            
            for upgrade in actions[-7:]:
                print(upgrade)
                self.upgradeLevel(upgrade)
        except:
            pass
        
        win, reward = self.checkWin()
        if win == False:
            for iceberg in self.iceBergs:
                if iceberg["Owner"] != -1:
                    iceberg["troops"] += (1 * iceberg["l"])
        
        self.checkTroopsReached()
        self.Turn += 1

        return reward, win, self.Turn

# Game
# sim = SimulationAI(10000, 10000)
# while True:
#     sim.sendTroops(0, 4, 2)
#     sim.play() # action => [ [[startId, endId, troopsNum] * x] , [upgradeId] ]
#     state = [[list(i.values()) for i in sim.iceBergs], [sim.Attacks]]
#     break