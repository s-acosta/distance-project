function out = make_drifting_grating_videos(varargin)
% MAKE_DRIFTING_GRATING_VIDEOS  Make one video per orientation and (optionally) frames for Teensy
% Example:
% out = make_drifting_grating_videos( ...
%   'orientations', 0:45:315, 'spat_freq', 0.004, 'temp_freq', 2, ...
%   'contrast', 1.0, 'phase0', 0, 'size', [240 240], ...
%   'fps', 60, 'pre_time', 1, 'on_time', 2, 'post_time', 1, ...
%   'outdir', 'stim_out', 'write_png_frames', true, 'write_rgb565_frames', true);

p = inputParser;
addParameter(p, 'orientations', 0:90:270);        % degrees, one video per entry
addParameter(p, 'spat_freq', 0.004);              % cycles/pixel
addParameter(p, 'temp_freq', 2.0);                % cycles/second (Hz)
addParameter(p, 'contrast', 1.0);                 % 0..1 Michelson, base luminance 0.5
addParameter(p, 'phase0', 0);                     % degrees
addParameter(p, 'size', [240 240]);               % [H W] pixels (match your TFT)
addParameter(p, 'fps', 60);                       % frames/sec
addParameter(p, 'on_time', 2.0);                  % seconds of drifting grating
addParameter(p, 'codec', 'MPEG-4');               % VideoWriter profile (or 'Motion JPEG AVI')
addParameter(p, 'quality', 95);                   % Video quality (0..100) for MPEG-4
addParameter(p, 'outdir', 'C:\Users\acostamendoza\projects\distance-project\hardware-control\disparity-device\stimulus');            % output folder
addParameter(p, 'basename', 'drift');             % base name for files
addParameter(p, 'write_png_frames', false);       % dump PNG per frame
addParameter(p, 'write_rgb565_frames', false);    % dump raw RGB565 per frame (row-major LE)
parse(p, varargin{:});
opt = p.Results;

H = opt.size(1); W = opt.size(2);
nOn   = round(opt.on_time   * opt.fps);
nTotal =  nOn;

% Phase increment per frame (degrees):
phaseStep = 360 * opt.temp_freq / opt.fps;

% Prepare output folder
if ~exist(opt.outdir,'dir'), mkdir(opt.outdir); end

% Helpers for RGB565 writing
rgb565_from_rgb = @(imgU8) ...
    bitor( bitshift(uint16(imgU8(:,:,1)),3), 0 ) + 0; %#ok<NASGU> (kept for readability)
to_rgb565 = @(img) rgb565_from_rgb_local(img);
to_payload = @(img) typecast( reshape( to_rgb565(img).', [], 1 ), 'uint8' );

videos = strings(1, numel(opt.orientations));
frameDirs = strings(1, numel(opt.orientations));
rgb565Dirs = strings(1, numel(opt.orientations));

for k = 1:numel(opt.orientations)
    theta = deg2rad(opt.orientations(k));
    cosT = cos(theta); sinT = sin(theta);

    % Coordinate grid centered at image center for cleaner rotation
    [xg, yg] = meshgrid( (0:W-1) - (W-1)/2, (0:H-1) - (H-1)/2 );

    % Projection along grating axis (in pixels)
    % cycles/pixel -> argument = 2*pi*sf*(x*cosT + y*sinT)
    proj = xg*cosT + yg*sinT;

    % Output names
    tag = sprintf('%s_ori%03d', opt.basename, round(mod(opt.orientations(k),360)));
    vidPath = fullfile(opt.outdir, sprintf('%s.mp4', tag));
    videos(k) = vidPath;

    writer = VideoWriter(vidPath, opt.codec);
    if strcmpi(opt.codec,'MPEG-4')
        writer.Quality = opt.quality;
        writer.FrameRate = opt.fps;
    else
        writer.FrameRate = opt.fps;
    end
    open(writer);

    if opt.write_png_frames
        fDir = fullfile(opt.outdir, sprintf('%s_frames_png', tag));
        if ~exist(fDir,'dir'), mkdir(fDir); end
        frameDirs(k) = fDir;
    end

    if opt.write_rgb565_frames
        rDir = fullfile(opt.outdir, sprintf('%s_frames_rgb565', tag));
        if ~exist(rDir,'dir'), mkdir(rDir); end
        rgb565Dirs(k) = rDir;

        % also drop a small README about the format
        fid = fopen(fullfile(rDir,'README.txt'),'w');
        fprintf(fid, ['Each file is one raw RGB565 frame, row-major LITTLE-endian, size %dx%d.\n' ...
                      'File name encodes frame index. Intended for Teensy: send header "IMG W H\\n" then the bytes.\n'], W, H);
        fclose(fid);
    end

    base = 0.5;
    gray = uint8(round(base * 255)) * ones(H, W, 'uint8');

    % ON (drifting)
    phase = opt.phase0; % degrees
    for i = 1:nOn
        % luminance in [0..1]: L = base + (contrast/2)*sin(2*pi*sf*proj + phase)
        arg = 2*pi*opt.spat_freq*proj + deg2rad(phase);
        L = base + (opt.contrast/2) * sin(arg);
        frame = uint8(round(255 * L));  % grayscale
        frame = repmat(frame, [1 1 3]); % 3 channels for RGB video / RGB565

        writeVideo(writer, frame);

        idx =i;
        if opt.write_png_frames
            imwrite(frame, fullfile(frameDirs(k), sprintf('frame_%06d.png', idx)));
        end
        if opt.write_rgb565_frames
            payload = to_payload(frame);
            fbin = fullfile(rgb565Dirs(k), sprintf('frame_%06d.bin', idx));
            fid = fopen(fbin,'w'); fwrite(fid, payload, 'uint8'); fclose(fid);
        end

        phase = phase + phaseStep;  % advance temporal phase
    end

    close(writer);
    fprintf('Wrote %s (%d frames @ %d fps)\n', vidPath, nTotal, opt.fps);

end

out = struct('videos',videos, 'png_dirs',frameDirs, 'rgb565_dirs',rgb565Dirs, ...
             'fps',opt.fps, 'size',[H W], 'nFrames',nTotal);

end

% ------- local helper: pack RGB888 -> RGB565 uint16 (HxW) -------
function rgb565 = rgb565_from_rgb_local(img)
% img: HxWx3 uint8
if ~isa(img,'uint8'), img = im2uint8(img); end
R5 = bitshift(uint16(img(:,:,1)), -3);
G6 = bitshift(uint16(img(:,:,2)), -2);
B5 = bitshift(uint16(img(:,:,3)), -3);
rgb565 = bitor( bitshift(R5,11), bitor(bitshift(G6,5), B5) );
end
