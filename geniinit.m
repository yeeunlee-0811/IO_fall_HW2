function [inan] = geniinit(inp)
% This function generates an index for every fist obs of each firm

np = inp.np;
nyr = inp.nyr;
pid = inp.pid;
yr = inp.yr;
n = inp.n;


mat = ones(n,1);

for i = 1:np
    for j = 1:nyr
        ipyr = pid ==i & yr == j;
        ipyrlag = pid ==i & yr == j-1;
        
        if sum(ipyrlag) == 0
            mat(ipyr,:) = NaN;
        end
    end
end

inan = isnan(mat(:,1));
end
