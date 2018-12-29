function [Y] = imagedB(grid_x, grid_y, Im, mindB, maxdB, no_hilbert)
%Returns abs(hilbert(Im)) in dB scale

    if (nargin == 6 && no_hilbert == 1)
        Y = abs(Im);
    else
        Y = abs(hilbert(Im));
    end
    Y = 20*log10(Y)-20*log10(max(max(Y)));
    
    %Clip Y below mindB and above maxdB
    Y(find(Y<mindB)) = mindB;
    Y(find(Y>maxdB)) = maxdB;
    
    %Map mindB-->0 and maxdB->255
    Y = (Y-mindB)*255/(maxdB-mindB);
    
    figure;
    image(grid_x, grid_y, Y);
    set(gca,'YDir','normal');
    colorbar;
    colormap(jet(256));
end

