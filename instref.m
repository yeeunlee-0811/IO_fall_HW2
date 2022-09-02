function [fininst,isr] = instref(Z)

nk = size(Z,2);

isr = zeros(nk,1);

z0 = Z(:,1:2);
for i=3:size(Z,2)
    zi = Z(:,i);
    reg = fitlm(z0,zi,'Intercept',false);
    if reg.Rsquared.Ordinary < 0.98
        isr(i,1) = i;
        if length(unique(zi))>1 % not constant
            zi = zi - mean(zi);
        end
        z0 = [z0 zi];
        % disp(i)
    end
end

fininst = z0;
end

