function calcPhospheneArea(subjID, session, data_dir, figures_dir)

% Set random seed
rng(42, 'twister');

data_path = [data_dir '/tmsRtnTpy_sub' subjID '_sess' session];
load(data_path);

% remove invalid trials and corresponding coil locations
tmsRtnTpy = remove_invalid_trials(tmsRtnTpy);

% idx_to_drop = find(tmsRtnTpy.Response.CoilLocation == 3);
% tmsRtnTpy.Response.CoilLocation(idx_to_drop) = [];
% tmsRtnTpy.Response.Detection(idx_to_drop) = [];
% tmsRtnTpy.Response.Drawing(idx_to_drop) = [];
% tmsRtnTpy.Response.CoilLocation(7:end) = tmsRtnTpy.Response.CoilLocation(7:end) - 1;
% Check for unique coil locations
LocInds = unique(tmsRtnTpy.Response.CoilLocation);
%LocInds = [2 3 4 5 6 7 8];
% Initialize PhosphReport
PhosphReport = struct;
PhosphReport = repmat(PhosphReport, [1, length(LocInds)]);

% Extract parameters from tmsRtnTpy
parameters = loadParameters(tmsRtnTpy);
% Not randomizing: Day 1 is always No TMS and day 2 is always TMS, 
% randomization now happens on the end of the experimenter
tms_timings = [1, 2, 3, 4]; 
VAs = zeros(length(LocInds), 1);
Targs = zeros(length(LocInds), 2);
more_drops = [];

%% Calculate border and area of each phosphene report
for coilInd = LocInds %1:length(LocInds)
    disp(['Testing coil Location: ' num2str(coilInd)])
    % list of trials for given CoilLocation with detections
    phsphTrials = find(tmsRtnTpy.Response.CoilLocation == coilInd & tmsRtnTpy.Response.Detection == 1);
    PhosphReport(coilInd).trialInd = phsphTrials;
    % compute area and border for each drawing for given CoilLocation
    for trial = 1:length(phsphTrials)
        trialInd = phsphTrials(trial);
        drawing = tmsRtnTpy.Response.Drawing{trialInd};
        poly_shape = rmholes(polyshape(drawing));
        PhosphReport(coilInd).polyshape{trial} = poly_shape;
    end
    
    % calculate the overlapping area across all phosphenes for this coilLocation
    overlap_polyshape = PhosphReport(coilInd).polyshape{1};
    for trial = 2:length(PhosphReport(coilInd).polyshape)
%         overlap_polyshape = intersect(overlap_polyshape, PhosphReport(coilInd).polyshape{trial});
          overlap_polyshape = union(overlap_polyshape, PhosphReport(coilInd).polyshape{trial});
    end
    overlap_polyshape = rmholes(overlap_polyshape);
    PhosphReport(coilInd).overlapPolyshape = overlap_polyshape;
    
    
    %% Calculate Target Sample-space
    [X_mean, Y_mean] = centroid(overlap_polyshape);
    if isnan(X_mean) || isnan(Y_mean)
        PhosphReport(coilInd).polyshape_centroid = [X_mean, Y_mean];
        PhosphReport(coilInd).coilHemifield = NaN;
        PhosphReport(coilInd).taskMap = NaN;
        more_drops = [more_drops, coilInd];
    else
        PhosphReport(coilInd).polyshape_centroid = [X_mean, Y_mean];
        % Get Radial limits of phosphenes    
        OverlapRadial_px = sqrt((overlap_polyshape.Vertices(:,1)-tmsRtnTpy.Params.screen.xCenter).^2 + ...
                           (overlap_polyshape.Vertices(:,2)-tmsRtnTpy.Params.screen.yCenter).^2);
        OverlapRadial_cm = OverlapRadial_px * tmsRtnTpy.Params.screen.pixSize;
        OverlapRadial_dva = atan2d(OverlapRadial_cm, tmsRtnTpy.Params.taskParams.viewingDistance);
        Radial_dva_quantiles = quantile(OverlapRadial_dva, 5);

        % compute buffer of r and polar angle of mean.
        [va, stimSize_px, targX, targY, nontargX, nontargY] = ...
            get_visang(X_mean, Y_mean, Radial_dva_quantiles, parameters);
        VAs(coilInd) = va;
        Targs(coilInd, :) = [targX, targY];

        % Store coilHemifield
        if X_mean > parameters.xCenter
            PhosphReport(coilInd).coilHemifield = 1; % Right visual field
        else
            PhosphReport(coilInd).coilHemifield = 2; % Left visual field
        end
        for sess = 1:parameters.sessions
            TMScond = tms_timings(sess);
            % Compute taskMaps
            for block = 1:parameters.numBlocks
                % Holder for orientations of target category and actual
                % target
                ori_target = repelem(parameters.valOris, parameters.numRepeatsPerBlock * 2);
                ori_target = ori_target(randperm(length(ori_target')));

                % actual target obtained by shuffling from a unfiorm distribution
                ori_target_ = arrayfun(@(x) (x-parameters.jitterOris)+(2*parameters.jitterOris*rand()), ori_target); 
                
                % Set the location of target and randomly decide if it will
                % be inside or outside PF on half the trials
                targIn = repmat([targX, targY], parameters.numTrials/2, 1);
                targOut = repmat([nontargX, nontargY], parameters.numTrials/2, 1);
                targSetOrig = [targIn; targOut];

                trialInds = randperm(parameters.numTrials);
                targSet = targSetOrig(trialInds', :);

                % 1 if stimulus is inside TMS VF
                stimVF = [ones(size(targIn, 1), 1); zeros(size(targOut, 1), 1)];
                stimVF = stimVF(trialInds);
                
                % Store all the details in PhosphReport
                PhosphReport(coilInd).taskMap(sess, block).TMScond = TMScond;
                PhosphReport(coilInd).taskMap(sess, block).stimVF = stimVF;
                PhosphReport(coilInd).taskMap(sess, block).targOri = ori_target;
                PhosphReport(coilInd).taskMap(sess, block).targOri_ = ori_target_;
                PhosphReport(coilInd).taskMap(sess, block).stimSize = stimSize_px;
                PhosphReport(coilInd).taskMap(sess, block).targX = targSet(:, 1);
                PhosphReport(coilInd).taskMap(sess, block).targY = targSet(:, 2);
            end
        end

        % Compute taskMaps for Practice
        for blockPractice = 1:parameters.numBlocksPractice
            % Holder for orientations of target category and actual
            % target
            ori_targetPractice = repelem(parameters.valOris, parameters.numRepeatsPerBlock * 2);
            ori_targetPractice = ori_targetPractice(randperm(length(ori_targetPractice')));

            % actual target obtained by shuffling from a unfiorm distribution
            ori_targetPractice_ = arrayfun(@(x) (x-parameters.jitterOris)+(2*parameters.jitterOris*rand()), ori_targetPractice);

            % Set the location of target and randomly decide if it will
            % be inside or outside PF on half the trials
            targInPractice = repmat([targX, targY], parameters.numTrials/2, 1);
            targOutPractice = repmat([nontargX, nontargY], parameters.numTrials/2, 1);
            targSetOrigPractice = [targInPractice; targOutPractice];

            trialIndsPractice = randperm(parameters.numTrials);
            targSetPractice = targSetOrigPractice(trialIndsPractice', :);

            % 1 if stimulus is inside TMS VF
            stimVFPractice = [ones(size(targInPractice, 1), 1); zeros(size(targOutPractice, 1), 1)];
            stimVFPractice = stimVFPractice(trialIndsPractice);

            % Store all the details in PhosphReport
            PhosphReport(coilInd).taskMapPractice(blockPractice).stimVF = stimVFPractice;
            PhosphReport(coilInd).taskMapPractice(blockPractice).targOri = ori_targetPractice;
            PhosphReport(coilInd).taskMapPractice(blockPractice).targOri_ = ori_targetPractice_;
            PhosphReport(coilInd).taskMapPractice(blockPractice).stimSize = stimSize_px;
            PhosphReport(coilInd).taskMapPractice(blockPractice).targX = targSetPractice(:, 1);
            PhosphReport(coilInd).taskMapPractice(blockPractice).targY = targSetPractice(:, 2);
        end
    end
end

% Eliminating the coil locations that have no overlapping phosphenes. This
% changes the indices of the coil locations again.
if ~isempty(more_drops)
    for tt = 1:length(more_drops)
        disp(['New coil location ' num2str(more_drops(tt)) ' has no overlapping phosphenes and has been eliminated.']);
        VAs(more_drops(tt)) = []; % removing the idx from VAs
        Targs(more_drops(tt), :) = []; % Removing the idx from targs
        PhosphReport(more_drops(tt)) = []; % removing the idx from PhosphReport
        LocInds(more_drops(tt)) = []; % removing the idx from the LocInds
        LocInds(more_drops(tt):end) = LocInds(more_drops(tt):end) - 1;
    end
end

%% Visualize Phosphene maps
fig1 = figure('visible', 'off');
fig1.Position = [50 50 1000 1000];
N = length(LocInds);
N1 = [1, 1, 2, 2, 2, 2, 3, 2, 3, 3, 3, 3];
N2 = [1, 2, 2, 2, 3, 3, 3, 4, 3, 4, 4, 4];
if N <= 12
    n1 = N1(N);
    n2 = N2(N);
else
    n1 = ceil(N/5);
    n2 = min([5 N]);
end

for coilInd = 1:N
    subplot(n1,n2,coilInd);
    plot(parameters.xCenter, parameters.yCenter,'+k', 'MarkerSize', 10, 'linewidth', 1.5);
    for trial = 1:length(PhosphReport(coilInd).polyshape)
        hold on;
        pg = plot(PhosphReport(coilInd).polyshape{trial}, 'linewidth', 1.5);
        pg.FaceColor = 'w';
        ax = gca; ax.YDir = 'reverse'; %axis off;
    end
    
    pg1 = plot(PhosphReport(coilInd).overlapPolyshape);
    pg1.FaceColor = 'k';
    ax = gca;
    ax.FontSize = 16;
    ax.XAxis.FontSize = 16;
    ax.YAxis.FontSize = 16;
    % Plot the centroid of phosphene
    plot([parameters.xCenter, PhosphReport(coilInd).polyshape_centroid(1)], ...
        [parameters.yCenter, PhosphReport(coilInd).polyshape_centroid(2)], 'm--', ...
        'MarkerSize', 10, 'linewidth', 1.5);
    plot(PhosphReport(coilInd).polyshape_centroid(1), PhosphReport(coilInd).polyshape_centroid(2), ...
        'g*', 'MarkerSize', 10, 'linewidth', 1.5);
    % Plot the target location chosen
    plot(Targs(coilInd, 1), Targs(coilInd, 2), 'r*', ...
        'MarkerSize', 10, 'linewidth', 1.5);
    xlim([0 parameters.screenXpixels]);
    ylim([0 parameters.screenYpixels]);
    xticks(parameters.screenXpixels * [0 1/4 1/2 3/4 1]);
    yticks(parameters.screenYpixels * [0 1/4 1/2 3/4 1]);
    hh = sgtitle('Overlapping Phosphenes');
    set(hh, 'FontSize', 24, 'FontWeight', 'bold');
    title({['Coil index: ' num2str(coilInd)], ['mean va: ' num2str(round(VAs(coilInd), 4))]}, 'FontSize', 20);
    pbaspect([1 1 1]);
end

%% Visualize stimulus space as a gabor
fig2 = figure('visible', 'off');
fig2.Position = [1000 1000 1000 1000];

for coilInd = 1:N
    subplot(n1,n2,coilInd);
    plotGrating(parameters, Targs(coilInd, 1), Targs(coilInd, 2));
    hold on;
    plot(PhosphReport(coilInd).polyshape_centroid(1), PhosphReport(coilInd).polyshape_centroid(2), ...
        'g*', 'MarkerSize', 10, 'linewidth', 1.5);
    plot(parameters.xCenter, parameters.yCenter,'+k', 'MarkerSize', 10, 'linewidth', 1.5);
    ax = gca; ax.YDir = 'reverse';
    ax.FontSize = 16;
    ax.XAxis.FontSize = 16;
    ax.YAxis.FontSize = 16;
    xlim([0 parameters.screenXpixels]);
    ylim([0 parameters.screenYpixels]);
    xticks(parameters.screenXpixels * [0 1/4 1/2 3/4 1]);
    yticks(parameters.screenYpixels * [0 1/4 1/2 3/4 1]);
    hh1 = sgtitle('Stimuli Sample Space');
    set(hh1, 'FontSize', 24, 'FontWeight', 'bold');
    title({['Coil index: ' num2str(coilInd)], ['mean va: ' num2str(round(VAs(coilInd), 4))]}, 'FontSize', 20);pbaspect([1 1 1]);
end

%% Save results
saveName_phosph = [data_dir '/PhospheneReport_sub' subjID '_sess' session];
save(saveName_phosph,'PhosphReport')

if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end
saveName_phosph_fig = [figures_dir '/PhospheneReport_sub' subjID '_sess' session];
saveName_samplespace_fig = [figures_dir '/SampleSpace_sub' subjID '_sess' session];

saveas(fig1,saveName_phosph_fig,'fig')
saveas(fig1,saveName_phosph_fig,'png')
saveas(fig1,saveName_phosph_fig,'epsc')
saveas(fig2,saveName_samplespace_fig,'fig')
saveas(fig2,saveName_samplespace_fig,'png')
saveas(fig2,saveName_samplespace_fig,'epsc')
end

