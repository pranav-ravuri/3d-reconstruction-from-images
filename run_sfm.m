function run_sfm(data_index)
    %clc
    %close all
    %clear

    addpath('data/')
    % Used for testing to get consistent results
    % data_index = 9;
    rng(42);

    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Prepare the data.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Load data
    [K, img_names, init_pair, threshold] = get_dataset_info(data_index);

    % For testing, can change what images are used and initial pair

    % Set thresholds
    epipolar_threshold = threshold/K(1 ,1);
    homography_threshold = 3*epipolar_threshold;
    translation_threshold = 15*epipolar_threshold;

    % Find and SIFT points and descirptors for all images
    [image_features, image_descriptors] = find_sift_points(img_names);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Step 1 & 2: Finding relative rotations (and translations) and
    % upgrading to absolute rotations and translations.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initalize space to save values.
    Rotations = cell(length(img_names));
    % Translations = cell(length(img_names));
    x_nor = cell(length(img_names));

    Rotations{1} = eye(3);

    % Loop through all image pairs, starting with 1 and 2.
    for loop_index = 1:length(img_names) - 1

        disp("First loop pair " + num2str(loop_index))

        % Find common points in the two images.
        [x_nor{loop_index}, ~] = find_image_points(image_features, image_descriptors, loop_index, loop_index + 1, K);
        
        % Parallel RANSAC
        E = E_and_H_RANSAC(x_nor{loop_index}{1}, x_nor{loop_index}{2}, epipolar_threshold, homography_threshold);
        [~, P] = estimate_P_and_select(x_nor{loop_index}, E);

        % Extract rotation from P.
        Rotations{loop_index+1} = P{2}(1:3, 1:3);
        % Translations{loop_index+1} = P{2}(:, 4);

        Rotations{loop_index+1} = Rotations{loop_index} * Rotations{loop_index+1};
        % Translations{loop_index+1} = -Rotations{loop_index+1}*Rotations{loop_index}'*Translations{loop_index} + Translations{loop_index+1};
    end

    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Step 3: Find 3D-points using the initial pair of images.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    im1_index = init_pair(1);
    im2_index = init_pair(2);

    % Find shared image points between the inital pair of images.
    [x_nor_init, desc_X] = find_image_points(image_features, image_descriptors, im1_index, im2_index, K);

    % Estimate E and eliminate outliers before estimating 3D-points.
    E = E_and_H_RANSAC(x_nor_init{1}, x_nor_init{2}, epipolar_threshold, homography_threshold);
    [x_nor_init, inliers] = eliminating_outliers(x_nor_init, E, epipolar_threshold);
    desc_X = desc_X(:, inliers);
    [X_est, ~] = estimate_P_and_select(x_nor_init, E);
    
    % Center and Rotate the points
    X_est_u = X_est - mean(X_est, 2);
    X_est_u = X_est_u(1:3, :);
    X_est_global = Rotations{im1_index}'*X_est_u;
    X_est_global(4, :) = 1;

    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Step 4: Looping over the images to find establish correspondences
    % between it and the 3D-points.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initialize cells to save values
    Projection_matrices = cell(1, length(img_names));

    % Loop through all images.
    for loop_index = 1: length(img_names)

        disp("Image " + num2str(loop_index))
    
        % Find matches between image and 3D-points.
        [matches, ~] = vl_ubcmatch(image_descriptors{loop_index}, desc_X);
        x = image_features{loop_index}(1:2, matches(1, :));
        X = X_est_global(:, matches(2, :));

        % Homogenize and normalize.
        x = [x; ones(1, size(x, 2))];
        x_nor = pflat(K\x);
    
        % Estimate T using a version of DLT
        T = estimate_T_robust(X(1:3, :), x_nor, Rotations{loop_index}, translation_threshold);

        % Update the P matrix.
        Projection_matrices{loop_index} = [Rotations{loop_index}, T];
    end
    disp("Loops done")

    %% 
    % Plotting all pairs of reconstructed points.
    figure
    
    x = [{}, {}];
    for i = 1:length(img_names) - 1

        % Select what pair to plot.
        test_pair = [i,i+1];

        % Find matches in that pair.
        [matches, ~] = vl_ubcmatch(image_descriptors{test_pair(1)}, image_descriptors{test_pair(2)});
        x1 = image_features{test_pair(1)}(1:2, matches(1, :));
        x2 = image_features{test_pair(2)}(1:2, matches(2, :));

        % Homogenize and normalize.
        x1 = [x1; ones(1, size(x1, 2))];
        x2 = [x2; ones(1, size(x2, 2))];
        x{1} = K\x1;
        x{2} = K\x2;

        % [x, ~] = eliminating_outliers(x, skew(Projection_matrices{i+1}(:, 4))*Rotations{i+1}, 10*epipolar_threshold);

        % Triangluate the 3D-points for the image pair
        Est_3D_Point = pflat(triangulate_3D_point_DLT({Projection_matrices{test_pair(1)}, Projection_matrices{test_pair(2)}}, x{1}, x{2}));

        % Eliminating outliers to far from the center of gravity.
        % Calculate the vector of distances
        distances = sqrt(sum((Est_3D_Point - mean(Est_3D_Point, 2)).^2, 1));

        % Calculate the 90% quantile of the distances
        quantile_90 = quantile(distances, 0.9);

        % Check the condition distance <= 5 * quantile_90 for each point
        condition_met = distances <= quantile_90;

        % Filter the points that satisfy the condition
        filtered_Xj = Est_3D_Point(:, condition_met);
        plot3(filtered_Xj(1, :), filtered_Xj(2, :), filtered_Xj(3, :), "*")
        hold on
        axis equal
    end
    plotcams(Projection_matrices);

end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section contains all functions used above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Takes a list of images and use SIFT to find
% the image features and descriptors for each image.
function [image_features, image_descriptors] = find_sift_points(img_names)
    num_images = length(img_names);
    image_features = cell(1, num_images);
    image_descriptors = cell(1, num_images);

    for i = 1:num_images
        im = imread(img_names{i});
        [image_features{i}, image_descriptors{i}] = vl_sift(single(rgb2gray(im)));
    end
end

% Finds the shared image points between the idx1 and idx2 images in
% image_descrptors and extracts the image points in image_features
% to return the homogenized and normalized image points.
% Also returns the descriptions of the matches in desc_X.
function [x, desc_X] = find_image_points(image_features, image_descriptors, idx1, idx2, K)
    
    matches = vl_ubcmatch(image_descriptors{idx1}, image_descriptors{idx2});
    x = [{}, {}];
    x1 = image_features{idx1}(1:2, matches(1, :));
    x2 = image_features{idx2}(1:2, matches(2, :));
    
    x1 = [x1; ones(1, size(x1, 2))];
    x2 = [x2; ones(1, size(x2, 2))];
    x{1} = K\x1;
    x{2} = K\x2;

    desc_X = image_descriptors{idx1}(:, matches(1, :));
end

% Estimates F using DLT.
function F = estimate_F_DLT(x1, x2)
    num_points = size(x1, 2);
    M = zeros(num_points, 9);
    for i = 1:num_points
        vec = x2(:, i)*x1(:, i)';
        M(i, :) = vec(:)';
    end
    
    [~, ~, V] = svd(M);

    v = V(:, end);
    F = reshape(v, [3 3]);
end

% Enforces rank deficiency constraint on values of E.
function E = enforce_essential(E)
    [U, ~, V] = svd(E);
    if det(U*V') < 0
        V = -V;
    end
    E = U*diag([1, 1, 0])*V';
end

% Extracts P from E and chooses the solution with most points
% in front of the cameras.
function [X, P] = estimate_P_and_select(x_nor, E)
    P1 = [eye(3), zeros(3, 1)];
    P2 = extract_P_from_E(E);
    P = [{}, {}];
    P{1} = P1;
    X_est = [{}, {}, {}, {}];
    z_point_infront = zeros(1, 4);
    for i = 1:4
        P{2} = P2{i};
        X_est{i} = pflat(triangulate_3D_point_DLT(P, x_nor{1}, x_nor{2}));
        x_proj1 = P{1}*X_est{i};
        x_proj2 = P{2}*X_est{i};

        in_front_1 = x_proj1(3, :) > 0;
        in_front_2 = x_proj2(3, :) > 0;

        z_point_infront(i) = sum(in_front_1.*in_front_2);
    end

    selected_p = find(z_point_infront==max(z_point_infront));
    X = X_est{selected_p};
    P{2} = P2{selected_p};
end

% Extract the four possibilites of P from E.
function P = extract_P_from_E(E)
    P = [{}, {}, {}, {}];
    [U, ~, V] = svd(E);
    if det(U*V') < 0
        V = -V;
    end
    W = [0, -1, 0;
         1, 0, 0;
         0, 0, 1];
    u3 = U(:, 3);
    P{1} = [U*W*V', u3];
    P{2} = [U*W*V', -u3];
    P{3} = [U*W'*V', u3];
    P{4} = [U*W'*V', -u3];
end

% Finds the inliers of x based on the epipolar errors.
% Returns the filtered x and the array of indices of which are inliers.
function [x, inliers] = eliminating_outliers(x, E, err_threshold)
    err_vec = (compute_epipolar_errors(E, x{1}, x{2}).^2 + compute_epipolar_errors(E', x{2}, x{1}).^2)/2;
    inliers = err_vec < err_threshold^2;
    x{1} = x{1}(:, inliers);
    x{2} = x{2}(:, inliers);
end

% Finds epipolar errors between x1s and x2s when using F.
% Function can also be used with normalized points and E.
function eu = compute_epipolar_errors(F, x1s, x2s)
    l = F*x1s;
    l = l./sqrt(repmat(l(1, :).^2 + l(2, :).^2, [3, 1]));
    eu = abs(sum(l.*x2s));
end

% Estimates T using RANSAC.
function T_best = estimate_T_robust(X, x, R, err_threshold)

    num_points = size(x, 2);
    
    alpha = 0.99;
    epsilon = 0.10;
    s = 2;
    itterations = ceil((log(1-alpha)/log(1-epsilon^s)));
    
    for i = 1:itterations
        perm = randperm(num_points);
        random_x = x(:, perm(1: 2));
        random_X = X(:, perm(1: 2));
        T = estimate_T_DLT(random_X, random_x, R);
        x_est = pflat(R*X + T);
        inliers = sqrt((x(1, :) - x_est(1, :)).^2 + (x(2, :) - x_est(2, :)).^2) < err_threshold;
        new_epsilon = sum(inliers)/num_points;
    
        if new_epsilon > epsilon
            epsilon = new_epsilon;
            itterations = ceil((log(1-alpha)/log(1-epsilon^s)));
            T_best = T;
        end
        if i >= itterations
            break
        end
    end
end

% Estimates translation T using a version of DLT.
function T = estimate_T_DLT(X, x, R)
    num_points = size(x, 2);
    A = zeros(2*num_points, 3);
    B = zeros(2*num_points, 1);

    X_r = R * X;
    A(1, :) = [1, 0, -x(1, 1)];
    A(2, :) = [0, 1, -x(2, 1)];
    A(3, :) = [1, 0, -x(1, 2)];
    A(4, :) = [0, 1, -x(2, 2)];
    
    B(1) = X_r(3, 1)*x(1, 1) - X_r(1, 1);
    B(2) = X_r(3, 1)*x(2, 1) - X_r(2, 1);
    B(3) = X_r(3, 2)*x(1, 2) - X_r(1, 2);
    B(4) = X_r(3, 2)*x(2, 2) - X_r(2, 2);
    
    T = A\B;
end

% Uses pairs of image points to triangulate a 3D-point.
function X_est = triangulate_3D_point_DLT(P, x1, x2)
    num_points = size(x1, 2);
    X_est = zeros(4, num_points);
    M = zeros(4, 4);
    for i = 1:num_points
        M(1, :) = P{1}(1, :) - x1(1, i)*P{1}(3, :);
        M(2, :) = P{1}(2, :) - x1(2, i)*P{1}(3, :);
        M(3, :) = P{2}(1, :) - x2(1, i)*P{2}(3, :);
        M(4, :) = P{2}(2, :) - x2(2, i)*P{2}(3, :);
        [~, ~, V] = svd(M);
        X_est(:, i) = V(:, end);
    end
end

% Plots the cameras in P as arrows.
function plotcams(P)
    c = zeros(4,length(P));
    v = zeros(3,length(P));
    for i = 1:length(P)
        c(:,i) = null(P{i});
        v(:,i) = P{i}(3,1:3);
    end
    c = c./repmat(c(4,:),[4 1]);
    quiver3(c(1,:),c(2,:),c(3,:),v(1,:), v(2,:), v(3,:),'-','LineWidth',1.5,'MaxHeadSize',1.5);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Below follows possibly useful functions but the related code is 
% commented out above as they worsen the result.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Estimates H using DLT.
function H = estimate_H_DLT(x1, x2)
    num_points = size(x1, 2);
    M = zeros(2*num_points, 9);

    for i = 1:num_points
        M(2*i - 1, 1:3) = x1(:, i)';  
        M(2*i - 1, 7:9) = -x2(1, i)*x1(:, i)';  

        M(2*i    , 4:6) = x1(:, i)';  
        M(2*i    , 7:9) = -x2(2, i)*x1(:, i)';  
    end
    [~, ~, V] = svd(M);
    v = V(:, end);
    H = reshape(v, [3 3])';
end

% Calculates a skew matrix from input v.
% Borrowed from Thanacha Choopojcharoen at:
% https://se.mathworks.com/matlabcentral/answers/
% 135281-how-to-span-a-3-1-vector-into-a-3-3-skew-symmetric-matrix
function S = skew(v)
    a = size(v,1)==3;
    n = size(v,round(1+a));
    V = permute(v,[round(2-a),3,round(1+a)]);
    I = repmat(eye(3),[1,1,n]);
    S = cross(repmat(V,[1,3]),I);
end


% Parallel RANSAC function to estimate E
function final_E = E_and_H_RANSAC(x1, x2, err_threshold_E, err_threshold_H)

    num_points = size(x1, 2);
    
    alpha = 0.99;
    epsilon_E = 0.10;
    epsilon_H = 0.10;
    s_E = 8;
    s_H = 4;
    iterations_E = ceil((log(1-alpha)/log(1-epsilon_E^s_E)));
    iterations_H = ceil((log(1-alpha)/log(1-epsilon_H^s_H)));

    iterations = max(iterations_E, iterations_H);
    
    for i = 1:iterations
        perm = randperm(size(x1, 2));
        random_x1 = x1(:, perm(1: 8));
        random_x2 = x2(:, perm(1: 8));

        E = enforce_essential(estimate_F_DLT(random_x1, random_x2));
        E = E./E(end, end);
        inliers = (compute_epipolar_errors(E, x1, x2).^2 + compute_epipolar_errors (E', x2, x1).^2) / 2 < err_threshold_E^2;
        new_epsilon_E = sum(inliers)/num_points;
    
        H = estimate_H_DLT(random_x1(:,1:4), random_x2(:,1:4));
        H = H./H(end, end);
        
        Hx = pflat(H*x1);
        inliers = (Hx(1,:) - x2(1,:)).^2 + (Hx(2,:) - x2(2,:).^2) < err_threshold_H^2;
        new_epsilon_H = sum(inliers)/num_points;

        if new_epsilon_E > epsilon_E
            epsilon_E = new_epsilon_E;
            iterations_E = ceil((log(1-alpha)/log(1-epsilon_E^s_E)));
            E_best = E;
        end
        
        if new_epsilon_H > epsilon_H
            [bool, E_from_H] = acceptable_E_from_H([{x1}, {x2}], H, err_threshold_E);
            if bool
                E_from_H = E_from_H./E_from_H(end, end);
                inliers = (compute_epipolar_errors(E_from_H, x1, x2).^2 + compute_epipolar_errors (E_from_H', x2, x1).^2) / 2 < err_threshold_E^2;
                new_epsilon_E_H = sum(inliers)/num_points;
            end

            if(bool) && (new_epsilon_E_H > epsilon_E)
                disp("Found a better H")
                epsilon_H = new_epsilon_H;
                epsilon_E = new_epsilon_E_H;
                iterations_H = ceil((log(1-alpha)/log(1-epsilon_H^s_H)));
                iterations_E =  ceil((log(1-alpha)/log(1-epsilon_E^s_E)));
                E_best = E_from_H;
            end
        end

        if i >= iterations_E
            disp("epsilon_E: " + epsilon_E)
            disp("epsilon_H: " + epsilon_H)
            final_E = E_best./E_best(end, end);
            break
        end
    end
end

% Check if it is acceptable and return the best one.
% E is acceptable if exactly one of the 4 cheirality configurations has
% all inliers (w.r.t. the epipolar constraint) in front of both cameras
function [bool, acceptable_E] = acceptable_E_from_H(x, H, err_threshold_E)
    [R1,t1, ~, R2,t2, ~, ~] = homography_to_RT(H);

    E1 = enforce_essential(skew(t1)*R1);
    E1 = E1./E1(end, end);
    E2 = enforce_essential(skew(t2)*R2);
    E2 = E2./E2(end, end);
    E = [{E1}, {E2}];
    isAcceptable = [0, 0];
    scores = [0, 0];
    for j = 1:2
        z_point_infront = zeros(1, 4);
        P1 = [eye(3), zeros(3, 1)];
        P2 = extract_P_from_E(E{j});
        P = [{}, {}];
        P{1} = P1;
        
        inliers = (compute_epipolar_errors(E{j}, x{1}, x{2}).^2 + compute_epipolar_errors (E{j}', x{2}, x{1}).^2) / 2 < err_threshold_E^2;
        filtered_x1 = x{1}(:, inliers);
        filtered_x2 = x{2}(:, inliers);
        scores(j) = sum(inliers);
        num_points = size(filtered_x1, 2);
        good_p = 0;
        for i = 1:4
            P{2} = P2{i};
            X_est = pflat(triangulate_3D_point_DLT(P, filtered_x1, filtered_x2));
            x_proj1 = P{1}*X_est;
            x_proj2 = P{2}*X_est;
    
            in_front_1 = x_proj1(3, :) > 0;
            in_front_2 = x_proj2(3, :) > 0;
    
            z_point_infront(i) = sum(in_front_1.*in_front_2);

            if z_point_infront(i) == num_points
                good_p = good_p+1;
            end
        end
        
        if good_p == 1
            isAcceptable(j) = 1;
        end
    end

    bool = true;
    if sum(isAcceptable) == 2
        if scores(1) > scores (2)
            acceptable_E = E{1};
        else
            acceptable_E = E{2};
        end
    elseif  isAcceptable(1) == 1
        acceptable_E = E{1};
    elseif  isAcceptable(2) == 1
        acceptable_E = E{2};
    else
        bool = false;
        acceptable_E = 1;
    end
end

% scales the points to make the last coordinate of a value 1
function projections = pflat(points)
    projections = points./points(end, :);
end