function [Ix Iy mag phase] = compute_image_derivatives(I, sigma, sigma0)

% FOR INTERNAL USE ONLY
% 0 -> traditional convolution
% 1 -> use IPP from matlab
mode = 1;

N_channels = size(I, 3);
N_std = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the differentiation kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kernel_ssize = ceil(N_std*sigma);
u_lin = -kernel_ssize:kernel_ssize;

% use separable kernels
Z = sqrt(2*pi)*sigma;

% compute the 1D differentiation kernels
sigma2 = sigma*sigma;
g = exp(-0.5*u_lin.^2/sigma2)/Z;
gx = -g.*u_lin/sigma2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the smoothing kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
smoothing = false;
if (nargin == 3) && ~isempty(sigma0)

    smoothing = true;

    kernel_ssize0 = ceil(N_std*sigma0);
    u_lin0 = -kernel_ssize0:kernel_ssize0;

    % use separable kernels
    Z0 = sqrt(2*pi)*sigma0;

    % compute the 1D differentiation kernels
    sigma02 = sigma0*sigma0;
    g0 = exp(-0.5*u_lin0.^2/sigma02)/Z0;

end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Type conversion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% convert the input image to take advantage of IPP
if ~strcmp(class(I), 'single')
    I = double(I);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Smoothing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if smoothing
    for h = 1:N_channels

        switch mode

            case 0

                I(:, :, h) = conv2(g0, g0, I(:,:,h), 'same');

            case 1

                I(:, :, h) = imfilter(I(:,:,h), g0(:), 'same', 'conv');
                I(:, :, h) = imfilter(I(:,:,h), g0, 'same', 'conv');

        end;

    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Differentiate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ix = zeros(size(I), 'single');
Iy = zeros(size(I), 'single');

for h = 1:N_channels

    switch mode

        case 0

            Ix(:, :, h) = conv2(gx, g, I(:,:,h), 'same');
            Iy(:, :, h) = conv2(g, gx, I(:,:,h), 'same');

        case 1

            Ix(:, :, h) = imfilter(I(:,:,h), gx(:), 'same', 'conv');
            Ix(:, :, h) = imfilter(Ix(:,:,h), g, 'same', 'conv');
            Iy(:, :, h) = imfilter(I(:,:,h), g(:), 'same', 'conv');
            Iy(:, :, h) = imfilter(Iy(:,:,h), gx, 'same', 'conv');

    end;

end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the magnitude
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargout > 2
    mag = zeros(size(I, 1), size(I, 2), 'single');
    for h = 1:N_channels
        tempx = immultiply(Ix(:, :, h), Ix(:, :, h));
        tempy = immultiply(Iy(:, :, h), Iy(:, :, h));
        tempM = imadd(tempx, tempy);
        mag = imadd(mag, tempM);
    end;
    mag = sqrt(mag);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the phase
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargout > 3
    phase = zeros(size(I, 1), size(I, 2), 'single');
    for h = 1:N_channels
        temp = single(atan2(Iy(:, :, h), Ix(:, :, h)));
        phase = imadd(phase, temp);
    end;
    phase = imdivide(phase, N_channels);
end

return
