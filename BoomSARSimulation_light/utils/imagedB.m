function [Y] = imagedB(grid_x, grid_y, Im, mindB, maxdB, no_hilbert)
%Returns abs(hilbert(Im)) in dB scale

    if (nargin == 6 && no_hilbert == 1)
        Y = abs(Im);
    else
        Y = abs(hilbert(Im));
    end
    Y = 20*log10(Y)-20*log10(max(max(Y)));
    
    %Clip Y below mindB and above maxdB
%     Y(find(Y<mindB)) = mindB;
%     Y(find(Y>maxdB)) = maxdB;

    imagesc(grid_x, grid_y, Y, [mindB, maxdB]);
    set(gca,'YDir','normal');
    colormap(jet(256));
  %  colorbar;
end

