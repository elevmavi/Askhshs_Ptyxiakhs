pkg load statistics; % Required for kmeans function

function cri = cjkcalc(idxTest, cidx)
    % This is a placeholder implementation, replace it with your actual calculation
    % You can use any similarity/distance metric calculation here
    cri = sum(abs(idxTest - cidx)); % Example calculation, replace it with actual calculation
endfunction

function [feature1, cri_last] = fss(xV, initidx, thres, nclust)
    replic = 100;
    CRItable = [];
    idxTest = initidx;
    selmescap = 1;
    fcols = columns(xV);

    for j = 1:fcols
        a = sum(xV(:, j));
        if isnan(a)
            CRItable = [CRItable; NaN, j, j];
        else
            cidx = kmeans(xV(:, j), nclust, 'MaxIter', replic, 'Replicates', replic);
            cri = cjkcalc(idxTest, cidx);
            CRItable = [CRItable; cri, j, j];
        endif
    endfor

    [crimax, mno] = max(CRItable(:, 1), [], 1);
    selmescap = CRItable(mno, 2);
    feature1 = num2str(selmescap);
    cri_last = crimax;

    while true
        for ai = 1:fcols
            a = sum(xV(:, ai));
            if mean(xV(:, ai)) <= 1e-4 || isnan(a)
                CRItable = [CRItable; NaN, ai, ai];
            else
                kmeanscols = [num2str(selmescap) ' ' num2str(ai)];
                kmeanscols = str2num(kmeanscols); % Convert string to numeric array
                fc = !ismember(ai, kmeanscols);
                if fc
                    cidx = kmeans(xV(:, kmeanscols), nclust, 'MaxIter', replic, 'Replicates', replic);
                    cri = cjkcalc(idxTest, cidx);
                    CRItable = [CRItable; cri, ai, ai];
                endif
            endif
        endfor

        [crimax, mno] = max(CRItable(:, 1), [], 1);
        selmescap = CRItable(mno, 2);
        q1 = (crimax - cri_last) / cri_last;

        if any(q1 > thres) && all(q1 >= 0.0)
            feature1 = [feature1 ', ' num2str(selmescap)];
            cri_last = crimax;
        endif

        if all(q1 < thres)
            break;
        endif
    endwhile
endfunction

% Load data from MATLAB file
data = load('resources/data/xV600x470.mat');
xV1 = data.xV1;

% Extract parameters
initidx = xV1(:, 1);
thres = 0.01;
nclust = 3;
xV = xV1(:, 2:end);

% Remove columns with NaN values
xV2 = xV(:, all(~isnan(xV)));

% Execute fss function
[feature1, mat] = fss(xV2, initidx, thres, nclust);

% Write results to a CSV file
csvwrite('fss_results.csv', [feature1, mat]);

