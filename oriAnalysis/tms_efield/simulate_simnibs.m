clear; close all; clc;
addpath("/users/mrugank/Applications/SimNIBS-4.0/matlab_tools/")

sub = 1;
subjID = num2str(sub, '%02d');
m2mfoldpath = ['/d/DATC/datc/MD_TMS_EEG/SIMNIBS_output/sub' ...
                subjID '/m2m_sub' subjID];
simfoldpath = ['/d/DATC/datc/MD_TMS_EEG/SIMNIBS_output/sub' ...
                subjID '/simnibs_simulation'];
S = sim_struct('SESSION');
S.fnamehead = [subjID '.msh'];
S.subpath = m2mfoldpath;
S.pathfem = simfoldpath;
S.open_in_gmsh=true;
S.map_to_vol=true;
S.tissues_in_niftis = 'all';
S.fields= 'eEjJvDs';
S.poslist{1} = sim_struct('TMSLIST');
S.poslist{1}.fnamecoil = 'MagVenture_Cool-B70.ccd';
S.poslist{1}.pos(1).centre = 'C3';
S.poslist{1}.pos(1).pos_ydir = 'CP3';
S.poslist{1}.pos(1).distance = 4;