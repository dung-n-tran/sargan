function [I_p] = backProject(Image, tx_x, tx_y, tx_z, rx_x, rx_y, rx_z, grid_x, grid_y, grid_z, fs, tAcq, oversample)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%BACKPROJECT : SAR back projection.
%
%Input parameters
%
%Image    - The raw image(num_apertures x length_of_pulse)
%Tx_range - Cordinates of Tx
%Rx_range - Cordinates of Rx
%grid_x,
%grid_y, 
%grid_z   - Grid points
%fs       - Sampling frequency
%tAcq     - Initial acquisition delay
%oversample - Oversampling factor. The raw image is interpolated by this
%             facor befor backprojection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[num_samples, num_ap] = size(Image); %Number of samples in the pulse

c = 3*10^8;

%Oversample the input image using fft
Image = interpft(Image, num_samples*oversample);

I_p = zeros(size(grid_x)); %Output image

%Do the grid all at once
for i = 1:num_ap

    %Distance from Tx and Rx to gridpoints
    r = sqrt((grid_x-tx_x(i)).^2+(grid_y-tx_y(i)).^2+(grid_z-tx_z(i)).^2)+...
        sqrt((grid_x-rx_x(i)).^2+(grid_y-rx_y(i)).^2+(grid_z-rx_z(i)).^2);
    t = r/c; %Time taken

    I = round((t-tAcq)*fs*oversample+1);
    idx = find(I<num_samples*oversample&I>0);

    I_p(idx) = I_p(idx) + reshape(Image(I(idx), i), size(I_p(idx)));
end


% %Do the grid a column at a time
% for i = 1:num_ap
%     tx_p = Tx_range(i, :);
%     rx_p = Rx_range(i, :);
%     
%     for j = 1:size(grid_x, 2)
%         g_x = grid_x(:, j);
%         g_y = grid_y(:, j);
%         g_z = grid_z(:, j);
% 
%         %Distance from Tx and Rx to gridpoints
%         r = sqrt((g_x-tx_p(1)).^2+(g_y-tx_p(2)).^2+(g_z-tx_p(3)).^2)+...
%             sqrt((g_x-rx_p(1)).^2+(g_y-rx_p(2)).^2+(g_z-rx_p(3)).^2);
%         t = r/c; %Time taken
%         
%         I = round((t-tAcq)*fs*oversample+1);
%         idx = find(I<num_samples*oversample&I>0);
%         
%         I_p(idx, j) = I_p(idx, j) + Image(I(idx), i);
%     end
% end


I_p = I_p/num_ap;


end
