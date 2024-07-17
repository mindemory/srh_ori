clear; close all; clc;

addpath(genpath('/usr/share/psychtoolbox-3'));

devId = PsychPowerMateMrugank('Open');
[button, dialPosOrig] = PsychPowerMateMrugank('Get', devId);

while true
    
    [button, dialPos] = PsychPowerMateMrugank('Get', devId);
    deltaDial = dialPos - dialPosOrig;
    currOri = rem(0 + 1 * deltaDial, 180)
end

PsychPowerMateMrugank('Close', devId);