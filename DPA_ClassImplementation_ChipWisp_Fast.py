# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:57:47 2022

@author: DREW
"""
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

sbox = sbox = [
                    [0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76],
                    [0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0],
                    [0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15],
                    [0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75],
                    [0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84],
                    [0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf],
                    [0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8],
                    [0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2],
                    [0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73],
                    [0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb],
                    [0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79],
                    [0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08],
                    [0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a],
                    [0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e],
                    [0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf],
                    [0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]
                    ]

class DPA:
    def __init__(self,traces,inputs,power_model):
        self.traces = traces
        self.inputs = inputs
        self.power_model = power_model
        
        self.NUM_KEYS = 256
        self.TRACE_LENGTH = self.traces.shape[1]
        self.NUM_PLAINTEXT = self.traces.shape[0]
        
    """UPDATE VALUES"""
    def update_values(self,new_traces,new_inputs):
        self.traces = np.append(self.traces,new_traces,axis = 0)
        self.inputs = np.append(self.inputs,new_inputs,axis = 0)
        """UPDATE CONSTANTS"""
        self.TRACE_LENGTH = self.traces.shape[0]
        self.NUM_PLAINTEXT = self.traces.shape[1]
        
    """PERFORM DPA"""
    def DPA(self):
        self.plain_text = self.arrange_plaintext(self.inputs)
        self.keys = self.generate_keys()
        self.round_key = self.generate_round_key(self.plain_text, self.keys)
        self.intermediate_value = self.generate_intermediate_value(self.round_key)
        self.intermediate_value_bin = np.array(self.decimal_to_binary(self.intermediate_value))
        if self.power_model == "HW":
            self.hamming_weight = self.hamming_weight(self.intermediate_value_bin)
            self.R = self.correlation_coefficient(self.traces, self.hamming_weight)
        if self.power_model == "HD":
            self.plain_text_bin = np.array(self.decimal_to_binary(self.plain_text))
            self.hamming_distance = self.hamming_distance(self.intermediate_value_bin,self.plain_text_bin)
            self.R = self.correlation_coefficient(self.traces, self.hamming_distance)
    
    """CONVERT DECIMAL TO BINARY"""
    def decimal_to_binary(self,array):
        binary_result = [ [0] * self.NUM_KEYS for _ in range(self.NUM_PLAINTEXT)]
        for i in range(len(binary_result)):
            for j in range(len(binary_result[i])):
                binary_result[i][j] = '{0:08b}'.format(array[i][j])
        return binary_result

    """ PLAIN TEXT """
    def arrange_plaintext(self,inputs):
        return np.repeat(inputs,self.NUM_KEYS,axis=1)
    
    """GENERATE KEYS ARRAY"""
    def generate_keys(self):
        keys = np.arange(self.NUM_KEYS)
        return np.tile(keys,(self.NUM_PLAINTEXT,1))
    
    """ GENERATE ROUND KEY """
    def generate_round_key(self,plaintext,keys):
        return plaintext ^ keys
    
    """ S-BOX & INTERMEDIATE VALUE """
    def generate_intermediate_value(self,roundkey):
        intermediate_value = [ [0] * self.NUM_KEYS for _ in range(self.NUM_PLAINTEXT)]
        intermediate_value = np.array(intermediate_value)
        """CONVERT INTERMEDIATE VALUE TO HEX AND PERFORM S-BOX"""
        for i in range(len(roundkey)):
            for j in range(len(roundkey[i])):
                byte1 = roundkey[i][j] & 15
                byte2 = roundkey[i][j] >> 4
                intermediate_value[i][j] = sbox[byte2][byte1]
        return intermediate_value

    """ HAMMING WEIGHT """
    def hamming_weight(self,intermediatevalue_bin):
        hamming_weight = [ [0] * self.NUM_KEYS for _ in range(self.NUM_PLAINTEXT)]
        hamming_weight = np.array(hamming_weight)
        for i in range(len(intermediatevalue_bin)):
            for j in range(len(intermediatevalue_bin[i])):
                total = 0
                for k in range(len(intermediatevalue_bin[i][j])):
                    total += np.intc(intermediatevalue_bin[i][j][k])
                hamming_weight[i][j] = total
        return hamming_weight
    
    """ HAMMING DISTANCE """
    def hamming_distance(self, intermediatevalue_bin, plaintext_bin):
        hamming_distance = [ [0] * self.NUM_KEYS for _ in range(self.NUM_PLAINTEXT)]
        hamming_distance = np.array(hamming_distance)
        for i in range(len(plaintext_bin)):
            for j in range(len(plaintext_bin[i])):
                total = 0
                for k in range(len(plaintext_bin[i][j])):
                    total += not(plaintext_bin[i][j][k]==intermediatevalue_bin[i][j][k])
                hamming_distance[i][j] = total
        return hamming_distance
    
    """ CORRELATION COEFFICIENT """
    def correlation_coefficient(self,traces,hamming_weight):
        R = abs(np.corrcoef(np.transpose(traces),np.transpose(hamming_weight)))
        R = R[self.TRACE_LENGTH:self.TRACE_LENGTH+self.NUM_KEYS,0:self.TRACE_LENGTH]
        return R
    
    """ PLOT CORRELATION vs Time """
    def plot_correlation(self):
        fig, ax = plt.subplots()
        for i in range(len(self.R)):
            ax.plot(np.arange(self.TRACE_LENGTH),self.R[i], linewidth=0.75, label=i)
        plt.xlabel("Time")
        plt.ylabel("Correlation Coefficient")
        plt.title("Correlation Coefficient using Hamming-Weight")
        
    """ PLOT CORRELATION vs Key"""
    def plot_correlation_key(self):
        fig, ax = plt.subplots()
        ax.plot(self.keys[0,:],np.amax(self.R,axis=1), linewidth=0.75)
        plt.xlabel("Keys")
        plt.ylabel("Correlation Coefficient")
        plt.title("Correlation Coefficient vs Keys (HW)")
    
    """ FIND KEY """ 
    def find_key(self):
        keyfind = []
        for i in range(len(self.R)):
            keyfind.append(max(self.R[i]))
        print("WINNING KEY: ",keyfind.index(max(keyfind)),"   ",max(keyfind))
        


class DPA_MTD:
    def __init__(self,traces,inputs, START_VAL, NUM_TRACES, power_model):
        self.traces = traces
        self.inputs = inputs
        self.power_model = power_model
        
        self.NUM_KEYS = 256
        self.NUM_TRACES = NUM_TRACES
        self.START_VAL = START_VAL
        
    """ PLOT CORRELATION VS NUMBER OF TRACES"""
    def plot_correlation_traces(self,BYTE):
        Points = [ [0.0] * self.NUM_TRACES for _ in range(self.NUM_KEYS)]
        Points = np.array(Points)
        
        for i in range(self.START_VAL,self.START_VAL+self.NUM_TRACES):
            DPA_Obj = DPA(traces[:i,:],inputs[:i,:],self.power_model)
            DPA_Obj.DPA()
            for j in range(self.NUM_KEYS):
                Points[j][i-self.START_VAL] = (np.amax(DPA_Obj.R,axis=1))[j]
        
        fig, ax = plt.subplots()
        for i in range(self.NUM_TRACES):
            ax.plot(range(self.NUM_TRACES),Points[i], linewidth=0.75, label=i)
        plt.xlabel("Number of Traces")
        plt.ylabel("Correlation Coefficient")
        plt.title("MTD Key Byte "+str(BYTE))


class TVLA:
    def __init__(self,fixed_traces,random_traces):
        self.fixed_traces = fixed_traces
        self.random_traces = random_traces
        
        self.NUM_TRACES = self.fixed_traces.shape[1]
        self.N_fixed = fixed_traces.shape[1]
        self.N_random = random_traces.shape[1]
    
    """ CALCULATE WELSH T FOR EACH SAMPLE """
    def calculate_TVLA(self):
        t =[]
        for i in range(self.NUM_TRACES):
            fixed_mean = np.average(self.fixed_traces[:,i])
            fixed_stddev = np.std(self.fixed_traces[:,i])
            random_mean = np.average(self.random_traces[:,i])
            random_stddev = np.std(self.random_traces[:,i])
            t.append((fixed_mean - random_mean)/math.sqrt(((fixed_stddev**2)/self.N_fixed)+((random_stddev**2)/self.N_random)))        
        self.t = np.array(t)
          
    """ PLOT TVLA """
    def plot_TVLA(self):
        fig, ax = plt.subplots()
        ax.plot(range(self.NUM_TRACES),self.t, linewidth=0.75, label=1)
        plt.xlabel("Time Samples")
        plt.ylabel("t")
        plt.title("Test Vector Leakage Assessment")

    

BASE_FILEPATH = r'D:\Carleton\SYSC\SYSC4907\ChipWisperer'
TRACES_FILEPATH = r'Power_Traces_Differet_Plaintext.csv'
INPUTS_FILEPATH = r'Different_Plaintext.csv'

BYTE = 0
traces = pd.read_csv(os.path.join(BASE_FILEPATH,TRACES_FILEPATH),header=None).to_numpy(np.double)
inputs = pd.read_csv(os.path.join(BASE_FILEPATH,INPUTS_FILEPATH),header=None).to_numpy(np.intc)[:,[BYTE]]

"""DPA Graph"""
DPA_Test = DPA(traces,inputs,"HW")
DPA_Test.DPA()
DPA_Test.plot_correlation()
DPA_Test.plot_correlation_key()
DPA_Test.find_key()

"""MTD Graph"""
NUM_TRACES = 196
START_VAL = 6

MTD_Test = DPA_MTD(traces, inputs, START_VAL, NUM_TRACES, "HW")
MTD_Test.plot_correlation_traces(BYTE)

""" Iterate MTD Graphs"""
for i in range(16):
    inputs = pd.read_csv(os.path.join(BASE_FILEPATH,INPUTS_FILEPATH),header=None).to_numpy(np.intc)[:,[i]]
    MTD_Test = DPA_MTD(traces, inputs, START_VAL, NUM_TRACES, "HW")
    MTD_Test.plot_correlation_traces(i)


FIXEDTRACES_FILEPATH = r'Power_Traces_Same_Plaintext.csv'
RANDOMTRACES_FILEPATH = r'Power_Traces_Differet_Plaintext.csv'

""" TVLA Graph"""
fixed_traces = pd.read_csv(os.path.join(BASE_FILEPATH,FIXEDTRACES_FILEPATH),header=None).to_numpy(np.double)
random_traces = pd.read_csv(os.path.join(BASE_FILEPATH,RANDOMTRACES_FILEPATH),header=None).to_numpy(np.double)
TVLA_Test = TVLA(fixed_traces,random_traces)
TVLA_Test.calculate_TVLA()
TVLA_Test.plot_TVLA()












