
function final_tubes = cut_path(paths)
alpha = 1;
final_tubes= actionPathSmoother4oneVideo(paths, alpha) ;

end

function final_tubes = actionPathSmoother4oneVideo(paths, alpha)
action_count =1;
final_tubes = struct('ts',[],'te',[],'label',[],'path_total_score',[],...
    'dpActionScore',[],'dpPathScore',[],'vid',[],...
    'path_boxes',cell(1),'path_scores',cell(1),'video_id',cell(1));


for p = 1 : length(paths) % taking top 3 paths
    M = [paths(p).allScore(:,1) .1*ones(size(paths(p).allScore(:,1)))]';

    [pred_path,time,D] = dpEM_max(M,alpha(1));
    [ Ts, Te, Scores,Label, DpPathScore] = extract_action(pred_path,time,D,1);
    for k = 1 : length(Ts)
        final_tubes.ts(action_count) = Ts(k);
        final_tubes.te(action_count) = Te(k);
        final_tubes.dpActionScore(action_count) = Scores(k);
        final_tubes.label(action_count) = Label(k);
        final_tubes.dpPathScore(action_count) = DpPathScore(k);
        final_tubes.path_total_score(action_count) = paths(p).total_score;
        final_tubes.path_boxes{action_count} = paths(p).boxes;
        final_tubes.path_scores{action_count} = paths(p).scores;
        action_count = action_count + 1;
    end

end
    
end


function [ts,te,scores,label,total_score] = extract_action(p,q,D,action)
action = 1;
indexs = find(p==action);

if isempty(indexs)
    ts = []; te = []; scores = []; label = []; total_score = [];
    
else
    indexs_diff = [indexs,indexs(end)+1] - [indexs(1)-2,indexs];
    ts = find(indexs_diff>1);
    
    if length(ts)>1
        te = [ts(2:end)-1,length(indexs)];
    else
        te = length(indexs);
    end
    ts = indexs(ts);
    te = indexs(te);
    scores = (D(action,q(te)) - D(action,q(ts)))./(te-ts);
    label = ones(length(ts),1)*action;
    total_score = ones(length(ts),1)*D(p(end),q(end))/length(p);
end
end

function [ts,te,scores,label,total_score] = extract_action_original(p,q,D,action)
% p(1:1) = 1;
indexs = find(p==action);

if isempty(indexs)
    ts = []; te = []; scores = []; label = []; total_score = [];
    
else
    indexs_diff = [indexs,indexs(end)+1] - [indexs(1)-2,indexs];
    ts = find(indexs_diff>1);
    
    if length(ts)>1
        te = [ts(2:end)-1,length(indexs)];
    else
        te = length(indexs);
    end
    ts = indexs(ts);
    te = indexs(te);
    scores = (D(action,q(te)) - D(action,q(ts)))./(te-ts);
    label = ones(length(ts),1)*action;
    total_score = ones(length(ts),1)*D(p(end),q(end))/length(p);
end
end
