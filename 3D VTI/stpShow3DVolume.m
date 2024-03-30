function stpShow3DVolume(data, dz,dx,dy, clim, xslices, yslices, zslices, horizons, varargin)
%% This function is used to display a 3D volume
%
% Inputs
%
% data : a 3d volume of size (sampNum*nInline*nCrossline), where sampNum is
% the number of sample points for each trace, nInline is the number of
% inline sections, nCrossline is the number of traces for each inline
% section. If the data is not organized as (sampNum*nInline*nCrossline),
% you can adjust the order by using function "permute".
%
% dt : the sample interval in microsecond.
%
% clim : which is used to set the colormap limits for the current axes. It
% should be a vector of size 1*2.
%
% xslices : the inline sections will be shown. The value should be a vector
% representing the inline numbers.
%
% yslices : the crossline sections will be shown. The value should be a vector
% representing the crossline numbers.
%
% zslices : the slices with the same time will be shown. The value should be a vector
% representing a series of time nodes.
%
% horizons : the horizons indicate the slices which not has the same time,
% so it need to indicate the time information of each horizon. It should be
% a array of structures, for each structure, it has .horizon of size
% (nInline*nCrossline) and .shift representing the shift time in
% microsecond.
%
% startTime : the time of the first sample point of this volume in
% microsecond. When startTime a scalar, it means each trace has the same
% starting time. Otherwise, it should be the size of (nInline*nCrossline).
%
% firstInlineId : the minimum inline number of the volume, the default
% value is 1.
%
% firstCrosslineId : the minimum crossline number of each inline section,
% the default value is 1.
%
% colormap : the colormap for showing the data
%
% attributeName : the attribute name of the data. For instance, it would be
% "Amplitude" for post-stack seismic data, or "Velocity (m/s)" for velocity data. 
%
% fontName : defualt is 'Times new roman'
%
% fontsize : default is 11.
% 
% fontweight : default is 'bold'.
%
% welllogs : the welllogs of the voume. It is a structure array. Each
% structrue records the information of one well, including .inline,
% .crossline, .data, .name, .color, .startTime
%
% showsize : the size of each shown well in the figure.
%
% filtCoef : the cutoff angle frequency of well-log data. The range of it is (0, 1]. 
% The default value is 1 which means no filtering is performed.
%
% isShading : whether the background of is white. The defaut value is
% false.
%
% Written by Bin She, University of Electronic Science and Technology of
% China, 01-31-2019 
% Email: bin.stepbystep@gmail.com
%

    load original_color.mat;
    
    [sampNum, nInline, nCrossline] = size(data);
    
    %% check inputs
    p = inputParser;
    
    addRequired(p, 'dt', @(x)(x>0));
    addRequired(p, 'clim', @(x) validateattributes(x,{'numeric'},{'size', [1, 2]}));
    
    addParameter(p, 'startTime', 0, @(x) (isscalar(x)&&(x>0)) ||  (size(x, 1) == nInline && size(x, 2) == nCrossline) );
    addParameter(p, 'firstInlineId', 1, @(x) (isscalar(x)&&(x>0)) );
    addParameter(p, 'firstCrosslineId', 1, @(x) (isscalar(x)&&(x>0)) );
    addParameter(p, 'colormap', original_color );
    addParameter(p, 'fontname', 'Times new roman' );
    addParameter(p, 'fontsize', 11 );
    addParameter(p, 'fontweight', 'bold' );
    addParameter(p, 'attributeName', 'V (km/s)' );
    addParameter(p, 'filtCoef', 1, @(x) ((isscalar(x)&&(x>0)&&(x<=1))));
    addParameter(p, 'welllogs', []);
    addParameter(p, 'showsize', 1, @(x) (isscalar(x)&&(x>0)));
    addParameter(p, 'isShading', 1, @(x) (islogical(x)));
    
    p.parse(dz, clim, varargin{:});  
    params = p.Results;
    
    if params.isShading
        params.colormap = [1 1 1; params.colormap];
    end

    %% calculate the wells which will be shown
    welllogs = params.welllogs;
%     wellData = [];
    wellPos = [];
    wellIndex = [];
    for i = 1 : length(welllogs)
        iwell = welllogs{i};
        
        if ismember(iwell.crossline, yslices) || ismember(iwell.inline, xslices)
            tmp = stpButtLowPassFilter(iwell.data, params.filtCoef);
            
            if( length(iwell.data) ~= sampNum)
               fprintf('The length of the well-log data must be the same as the data volume!!!\n');
               return;
            end
            
            ix = iwell.inline - params.firstInlineId + 1;
            iy = iwell.crossline - params.firstCrosslineId + 1;
            
%             wellData = [wellData, tmp];
            wellPos = [wellPos; [ix, iy]];
            wellIndex = [wellIndex; i];
            
            showsize = params.showsize;
            
            for m = ix-showsize:ix+showsize-1
                for n = iy-showsize:iy+showsize-1
                    data(:, m, n) = tmp;
                end
            end
        end
    end
    
    
    %% prepare basical information
    if isscalar(params.startTime)
        minTime = params.startTime;
        newSampNum = sampNum;
        maxTime = minTime + sampNum * dz;
        V = data;
    else
%         minTime = min(min(startTime));
        % create horizons
        [V, minTime, maxTime, newSampNum] = stpHorizontalData(data, params.startTime, dz);
    end
   
    % time information
    t = minTime : dz : minTime + (newSampNum-1)*dz;
    
    firstInline = params.firstInlineId;
    firstCrossline = params.firstCrosslineId;
    endInline = firstInline + nInline - 1;
    endCrossline = firstCrossline + nCrossline - 1;
    
    [x, y, z] = meshgrid(firstInline *dx:dx: endInline*dx, firstCrossline *dy:dy: endCrossline*dy, t);
    
    %% calculate the horizons
    nHorizon = length(horizons);
    zHorizons = cell(1, nHorizon);
    
    for iHorizon = 1 : nHorizon
        zh.xd = zeros(nInline, nCrossline);
        zh.yd = zeros(nInline, nCrossline);
        zh.zd = zeros(nInline, nCrossline);
        
        for i = firstInline : endInline
            for j = firstCrossline : endCrossline
                zh.xd(i, j) = i;
                zh.yd(i, j) = j;
                zh.zd(i, j) = (horizons(iHorizon).horizon(i, j) + horizons(iHorizon).shift);
            end
        end
        zHorizons{iHorizon} = zh;
    end
    
    %% start to plot the data
    p_V = permute(V, [3 2 1]);

    houts = slice(x, y, z, p_V, xslices, yslices, zslices, 'cubic'); hold on;
    horizonSlices = cell(1, nHorizon);
    
    for iHorizon = (1 : nHorizon)*25
        horizonSlices{iHorizon} = slice(x, y, z, p_V, zHorizons{iHorizon}.xd, zHorizons{iHorizon}.yd, zHorizons{iHorizon}.zd, 'cubic');
    end
    %% show the titles of all crossed well
    for i = 1 : size(wellPos, 1)
        ix = wellPos(i, 1);
        iy = wellPos(i, 2);

        if isscalar(params.startTime)
            text(ix, iy, minTime - 0.02*(maxTime-minTime), welllogs{wellIndex(i)}.name, ...
            'color', welllogs{wellIndex(i)}.color, 'fontsize', params.fontsize,...
            'fontweight','bold', 'fontname', params.fontname);
        else
            text(ix, iy, params.startTime(ix, iy) - 0.02*(maxTime-minTime), welllogs{wellIndex(i)}.name, ...
            'color', welllogs{wellIndex(i)}.color, 'fontsize', params.fontsize,...
            'fontweight','bold', 'fontname', params.fontname);
        end
    end
    
    annotation(gcf,'textbox',[0.52 0.058 0.33 0.05],...
        'String',{params.attributeName},...
        'LineStyle','none',...
        'FontWeight', params.fontweight,...
        'FontSize',  params.fontsize,...
        'FontName', params.fontname,...
        'FitBoxToText','off', 'EdgeColor',[0.94 0.94 0.94]);
    
    
    
    if params.isShading
        shading interp;
    else
        shading flat;
    end
    
    for i = 1 : length(houts)
        stpChangeColorData(houts(i), params.isShading, clim, params.colormap);
    end
    
    for iHorizon = 1 : nHorizon
        stpChangeColorData(horizonSlices{iHorizon}, params.isShading, clim, params.colormap);
    end
    
%     shading interp;
    
    set(gca, 'clim', clim);
    colormap(params.colormap);
    
    
    colorbar('location', 'southoutside', 'position', [0.092 0.039 0.88 0.02]);
    
    set(gca,'zdir','reverse');
    xlabel('X (km)', 'fontsize', params.fontsize,'fontweight', params.fontweight, 'fontname', params.fontname);
    ylabel('Y (km)', 'fontsize', params.fontsize,'fontweight', params.fontweight, 'fontname', params.fontname);
    zlabel('Z (km)', 'fontsize', params.fontsize,'fontweight', params.fontweight, 'fontname', params.fontname);
    set(gca , 'fontsize', params.fontsize,'fontweight', params.fontweight, 'fontname', params.fontname);    
    
    
    
end

function stpChangeColorData(isurface, isShading, clim, colormap)

    if isShading
        nColor = size(colormap, 1);
        cinterval = (clim(2) - clim(1)) / nColor; 
        newLmin = clim(1) + 2*cinterval;
        d1 = isurface.CData;

        d1( find((d1 ~= inf) & (d1 < newLmin)) ) = newLmin;
        set( isurface, 'CData', d1);
    else
        set(isurface, 'AlphaDataMapping', 'direct');
        set(isurface, 'AlphaData', isurface.CData ~= 0);
        set(isurface, 'FaceColor', 'texturemap');
        set(isurface, 'FaceAlpha', 'texturemap');
    end
    

end

function [V, minTime, maxTime, newSampNum] = stpHorizontalData(data, startTime, dt)
    
    [sampNum, nInline, nCrossline] = size(data);
    
    minTime = floor((min(startTime(:)) - 2*dt)/10)*10;
    maxTime = max(startTime(:)) + 2*dt + sampNum * dt;
    
    newSampNum = round( (maxTime - minTime) / dt );
    V = zeros(newSampNum, nInline, nCrossline);
    V(:) = inf;
    
    for i = 1 : nInline
        for j = 1 : nCrossline
            
            index = round( (startTime(i, j) - minTime) / dt );  
            
            V(index+1 : index+sampNum, i, j) = data(:, i, j);
        end
    end
end