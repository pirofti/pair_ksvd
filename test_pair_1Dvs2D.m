% Copyright (c) 2018-2019 Paul Irofti <paul@irofti.net>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
% 
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

% Cite as:
% P. Irofti and B. Dumitrescu, “Pairwise Approximate K-SVD,” 
% in Acoustics Speech and Signal Processing (ICASSP), 
% 2019 IEEE International Conference on, 2019, pp. 1--5.


%% Separable: Dictionary learning 1D versus 2D
clear; clc; close all; fclose all; format compact;
%%-------------------------------------------------------------------------
p = 8;                  % patch size
s = 6;                  % sparsity
N = 4000;               % total number of patches
n = 256;                % dictionary size
nfactor = [4 6 8];      % 2D atoms: nfactor(i)*sqrt(n)
iters = 500;            % DL iterations
replatom = 'worst';     % replace unused atoms
rounds = 1;             % rounds

curves = {'2D', '1D'};  % plot curve names
color = ['g', 'r'];     % plot curve colors
%%-------------------------------------------------------------------------
updates = {'pair_dl', 'aksvd'};
spfuncs = {'pair_omp', 'omp'};
methods = [
  % Name	Function         Dictionary index
  {'ak2D', @denoise_2D,      1};
  {'ak1D', @denoise_omp,     2};
];
%%-------------------------------------------------------------------------
datadir = 'data\';
dataprefix = 'pair_1Dvs2D';

imdir = 'img\';
img_train = {'lena.png', 'barbara.png', 'boat.png', 'peppers.png' 'house.png',};
%%-------------------------------------------------------------------------
addpath(genpath('DL')); % Set to your local copy of dl-box
ts = datestr(now, 'yyyymmddHHMMss');
%%-------------------------------------------------------------------------
% EXPERIMENTS
%%-------------------------------------------------------------------------
for nf = nfactor
    fprintf('nf=%d: ', nf);
    [n1, n2] = deal(nf*sqrt(n));
%%-------------------------------------------------------------------------
% INITIALIZATION
%%-------------------------------------------------------------------------
    ups = length(updates);
    Dall = cell(ups,1);
    Xall = cell(ups,1);
    D0 = cell(ups,1);
    D0{2} = odctdict(p^2,n);
    D0{1} = {odctdict(p,n1) odctdict(n2,p)};
%%-------------------------------------------------------------------------
% LEARNING
%%-------------------------------------------------------------------------
    Dall = cell(rounds,ups,1);
    Xtrainall = cell(rounds,ups,1);
    errsall = zeros(rounds,ups,iters);
    Y = [];
    Ytrain = cell(rounds,1);
    for iimg = 1:length(img_train)
        img = img_train{iimg};
        I = double(imread([imdir,char(img)]));
        I = I(:, :, 1);
        Y = [Y im2col(I, [p p], 'sliding')];
    end
    for r = 1:rounds
        Ytrain{r} = Y(:,randperm(size(Y,2), N));
    end
    clear Y;

    for r = 1:rounds
        fprintf('%d', mod(r,rounds));
        for i = 1:ups
            fprintf('%s', updates{i}(1));
            [Dall{r}{i}, Xtrainall{r}{i}, errsall(r,i,:)] = ...
                DL(Ytrain{r}, D0{i}, s, iters, ...
                str2func(updates{i}), 'spfunc', str2func(spfuncs{i}), ...
                'replatom', replatom);
        end
    end
    fname = [datadir dataprefix '-dictionaries' ...
        '-n' num2str(n) '-nf'  num2str(nf) '-N' num2str(N) '-' ...
        '-K' num2str(iters) '-r' num2str(rounds) ...
        ts '.mat'];
    save(fname, 'Ytrain', 'Dall', 'errsall');      
%%-------------------------------------------------------------------------
% PLOT DATA
%%-------------------------------------------------------------------------
    figure(nf);
    rmsedata = squeeze(mean(errsall,1));
    for i = 1:ups
        plot(1:iters, rmsedata(i,:), color(i));
        hold on;
    end
    hold off;
    lgd = legend(curves);
    title(['nf=' num2str(nf)]);
end