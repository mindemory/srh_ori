function tmsRtnTpy = remove_invalid_trials(tmsRtnTpy)
% The function has been added to remove trials wherein the subject detects
% phosphenes but accidently ends drawing phosphenes too early.

total_trials = size(tmsRtnTpy.Response.Detection, 2);
invalid_trials = [];

% Invalid trials are the ones that do not have NaN in the drawing
% coords but have fewer data points. Here less than 100 data points is
% considered as an invalid trial.
for tt = 1:total_trials
    if ~sum(isnan(tmsRtnTpy.Response.Drawing{tt}), 'all') && ...
            size(tmsRtnTpy.Response.Drawing{tt}, 1) < 100
        invalid_trials = [invalid_trials, tt];
    end
end

% Replacing invalid trials by undetected trials
for trial = invalid_trials
    tmsRtnTpy.Response.Detection(trial) = 0;
    tmsRtnTpy.Response.Drawing{trial} = NaN;
end

% Removing coil locations with 1 or no detections
for coilLocInd = unique(tmsRtnTpy.Response.CoilLocation)
    inds = tmsRtnTpy.Response.CoilLocation == coilLocInd;
    detections = tmsRtnTpy.Response.Detection(inds);
    if sum (detections) <= 1
        disp(['Coil location ', int2str(coilLocInd), ' has 1 or no detections and will be eliminated.'])
        tmsRtnTpy.Response.CoilLocation(inds) = [];
        tmsRtnTpy.Response.Detection(inds) = [];
        tmsRtnTpy.Response.Drawing(inds) = [];
    end
end

% Renumber CoilLocations
unique_coilLocs = unique(tmsRtnTpy.Response.CoilLocation);
for locInd = 1:length(unique_coilLocs)
    coilLocInd = unique_coilLocs(locInd);
    inds = tmsRtnTpy.Response.CoilLocation == coilLocInd;
    tmsRtnTpy.Response.CoilLocation(inds) = locInd;
end
end