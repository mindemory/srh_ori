function calcPhospheneArea_biasreplication(subjID, session, coilLocInd, data_dir)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bias-replication variant of calcPhospheneArea.m.
%
% IMPORTANT: This does NOT recompute phosphene localization. It assumes
% the subject already has a valid PhospheneReport_sub<ID>_sess<N>.mat on
% disk (produced by the ORIGINAL calcPhospheneArea.m pipeline) and simply
% reuses the already-measured target/non-target screen coordinates and
% stimulus size for the given coilLocInd, then builds a fresh trial/task
% structure around the bias-replication design:
%   - target orientations fixed to 45/135 deg
%   - jitter is discrete, +/-1 to +/-24 deg in integer steps (48 values)
%   - only No-TMS and Middle-TMS conditions (8 blocks each, 24 trials/block)
%   - each (orientation x RF-location) combination gets all 48 jitter
%     values exactly once across a TMS condition's 8 blocks
%
% subjID, session:  subject/session identifiers (numeric or string), same
%                    convention as S01_phosphene_analysis / S02_saveTaskMap
% coilLocInd:        coil location index already established for this
%                    subject (same index passed to S02_saveTaskMap)
% data_dir:          path to the phosphene_data directory containing the
%                    subject's original PhospheneReport_sub<ID>_sess<N>.mat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(42, 'twister');

subjID = num2str(subjID, "%02d");
session = num2str(session, "%02d");

%% Load the ORIGINAL phosphene report for this subject/session (never overwritten)
orig_path = [data_dir '/PhospheneReport_sub' subjID '_sess' session];
S_orig = load(orig_path); % expects variable PhosphReport inside
PhosphReport_orig = S_orig.PhosphReport;

if coilLocInd > length(PhosphReport_orig) || isempty(PhosphReport_orig(coilLocInd).taskMap) || ...
        (isnumeric(PhosphReport_orig(coilLocInd).taskMap) && all(isnan(PhosphReport_orig(coilLocInd).taskMap(:))))
    error(['Coil location ' num2str(coilLocInd) ' has no valid phosphene taskMap in ' orig_path]);
end

% Pull the already-measured in-PF / out-PF coordinates and stimulus size
% for this coil location. targX/targY in the original taskMap are
% per-trial (already resolved to the in- or out-PF coordinate depending
% on that trial's stimVF), so recover the two distinct coordinates from
% trials of each type.
tb = PhosphReport_orig(coilLocInd).taskMap(1, 1);
inIdx = find(tb.stimVF == 1, 1);
outIdx = find(tb.stimVF == 0, 1);
targX = tb.targX(inIdx);       targY = tb.targY(inIdx);
nontargX = tb.targX(outIdx);   nontargY = tb.targY(outIdx);
stimSize_px = tb.stimSize;

%% Build bias-replication trial structure
bparams = loadParameters_biasreplication();

PhosphReport = struct;
PhosphReport = repmat(PhosphReport, [1, coilLocInd]);

% -- Main blocks: taskMap(1,:) = No-TMS (8 blocks), taskMap(2,:) = Middle-TMS (8 blocks)
for c = 1:length(bparams.tmsConds)
    TMScond = bparams.tmsConds(c);

    % Exhaustive trial set for this TMS condition: each (orientation x
    % RF-location) combination gets all 48 jitter values exactly once.
    nTotal = bparams.numTrialsPerBlock * bparams.numBlocksPerCond; % 192
    trialList = zeros(nTotal, 3); % columns: [targOri, targOri_, stimVF]
    rowIdx = 1;
    for oi = 1:length(bparams.valOris)
        thisOri = bparams.valOris(oi);
        for rf = [1, 0] % 1 = in-PF, 0 = out-PF
            jitterShuf = bparams.jitterVals(randperm(bparams.numJitters));
            for jj = 1:bparams.numJitters
                trialList(rowIdx, :) = [thisOri, thisOri + jitterShuf(jj), rf];
                rowIdx = rowIdx + 1;
            end
        end
    end
    trialList = trialList(randperm(size(trialList, 1)), :); % shuffle overall trial order

    for block = 1:bparams.numBlocksPerCond
        rows = (block-1)*bparams.numTrialsPerBlock + 1 : block*bparams.numTrialsPerBlock;
        blockTrials = trialList(rows, :);

        inMask = blockTrials(:, 3) == 1;
        targX_block = zeros(size(blockTrials, 1), 1);
        targY_block = zeros(size(blockTrials, 1), 1);
        targX_block(inMask) = targX;    targY_block(inMask) = targY;
        targX_block(~inMask) = nontargX; targY_block(~inMask) = nontargY;

        PhosphReport(coilLocInd).taskMap(c, block).TMScond = TMScond;
        PhosphReport(coilLocInd).taskMap(c, block).stimVF = blockTrials(:, 3);
        PhosphReport(coilLocInd).taskMap(c, block).targOri = blockTrials(:, 1);
        PhosphReport(coilLocInd).taskMap(c, block).targOri_ = blockTrials(:, 2);
        PhosphReport(coilLocInd).taskMap(c, block).stimSize = stimSize_px;
        PhosphReport(coilLocInd).taskMap(c, block).targX = targX_block;
        PhosphReport(coilLocInd).taskMap(c, block).targY = targY_block;
    end
end

% -- Practice blocks: same 45/135 orientations + integer jitter, but drawn
% with replacement (not required to be exhaustive, matching the casual
% role practice blocks played in the original protocol). TMScond is
% irrelevant for practice -- ori_task_biasreplication.m always forces
% No-TMS during practice regardless of any field here.
for blockP = 1:bparams.numBlocksPractice
    oriCat = bparams.valOris(randi(length(bparams.valOris), bparams.numTrialsPerBlock, 1))';
    jitterDraw = bparams.jitterVals(randi(bparams.numJitters, bparams.numTrialsPerBlock, 1))';
    oriActual = oriCat + jitterDraw;

    stimVFp = [ones(bparams.numTrialsPerBlock/2, 1); zeros(bparams.numTrialsPerBlock/2, 1)];
    stimVFp = stimVFp(randperm(bparams.numTrialsPerBlock));

    inMaskP = stimVFp == 1;
    targXp = zeros(bparams.numTrialsPerBlock, 1);
    targYp = zeros(bparams.numTrialsPerBlock, 1);
    targXp(inMaskP) = targX;    targYp(inMaskP) = targY;
    targXp(~inMaskP) = nontargX; targYp(~inMaskP) = nontargY;

    PhosphReport(coilLocInd).taskMapPractice(blockP).stimVF = stimVFp;
    PhosphReport(coilLocInd).taskMapPractice(blockP).targOri = oriCat;
    PhosphReport(coilLocInd).taskMapPractice(blockP).targOri_ = oriActual;
    PhosphReport(coilLocInd).taskMapPractice(blockP).stimSize = stimSize_px;
    PhosphReport(coilLocInd).taskMapPractice(blockP).targX = targXp;
    PhosphReport(coilLocInd).taskMapPractice(blockP).targY = targYp;
end

%% Save results (new filename prefix -- never collides with the original report)
saveName_phosph = [data_dir '/PhospheneReport_biasrep_sub' subjID '_sess' session];
save(saveName_phosph, 'PhosphReport');
end
