function drawBBoxes( boxes, varargin )
ip = inputParser;
ip.addParamValue('LineWidth',  1, @isscalar);
ip.addParamValue('LineStyle', '-', @ischar);
ip.addParamValue('EdgeColor', 'b', @ischar);
ip.addParamValue('Scores',   [], @isnumeric);
ip.addParamValue('Sname',    '%.2f', @ischar);
ip.addParamValue('FontSize', 8, @isscalar);
ip.addParamValue('BackgroundColor', 'w', @ischar);

ip.parse(varargin{:});
opts = ip.Results;

if ~isempty(boxes)
    draw_positions = [boxes(:,1),boxes(:,2),boxes(:,3)-boxes(:,1)+1, boxes(:,4)-boxes(:,2)+1];
    for b = 1:size(draw_positions,1) 
        rectangle('Position', draw_positions(b,:), 'LineWidth', opts.LineWidth, 'EdgeColor', opts.EdgeColor, 'LineStyle', opts.LineStyle);
    end
    if ~isempty(opts.Scores)
        assert(numel(opts.Scores) == size(boxes,1))
        for b = 1:size(draw_positions,1) 
            label = sprintf(opts.Sname, opts.Scores(b));
            text(double(boxes(b,1))+2, double(boxes(b,2)),label,...
                'BackgroundColor', opts.BackgroundColor, 'FontSize', opts.FontSize);
        end 
    end
end

end