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


%% Test pair DL
clear; clc; close all; fclose all; format compact;
%%-------------------------------------------------------------------------
N = 100;   % total number of patches
p = 8;      % patch size
n1 = 50;    % atoms in left dictionary
n2 = 50;    % atoms in right dictionary
s = 4;      % sparsity
iters = 50; % DL iterations
%%-------------------------------------------------------------------------
addpath(genpath('DL'));  % Set to your local copy of dl-box

Y = randn(p^2,N);
D1 = normc(randn(p,n1));
D2 = normr(randn(n2,p));

[D,X,errs,extraerrs] = DL(Y, {D1,D2}, s, iters, ...
    str2func('pair_dl'), 'spfunc', str2func('pair_omp'));

disp(['2D AK-SVD ' num2str(min(errs))]);
plot(1:iters, errs);