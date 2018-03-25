
%perceptron using the MNIST dataset and evaluate its performance.
clear all;
    % Load MNIST.
    images = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');
   
    
    % Transform the labels to Binary format.
    labels1 = full(ind2vec(labels' + 1)) ;
    hidden_units = 300;
    learn_rate = 0.3;
    
    % Choose activation function.
    act_fn = @logisticSigmoid;
    dact_fn = @dLogisticSigmoid;
    
    batch_size = 100;
    epochs = 500;
    train_size = size(images, 2);
    input_dim = size(images, 1);

    output_dim = size(labels1, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hid_wt = rand(hidden_units, input_dim);
    out_wt = rand(output_dim, hidden_units);
    
    hid_wt = hid_wt./size(hid_wt, 2);
    out_wt = out_wt./size(out_wt, 2);
    
    n = zeros(batch_size);
    
   
    for t = 1: epochs
        for k = 1: batch_size
            % Select which input vector to train on.
            n(k) = floor(rand(1)*train_size + 1);
            
            % Propagate the input vector through the network.
            input_vec = images(:, n(k));
            hid_input = hid_wt*input_vec;
            hid_vec_out = act_fn(hid_input);
            out_actual = out_wt*hid_vec_out;
            out_vec1 = act_fn(out_actual);
            
            tar_vec1 = labels1(:, n(k));
            
            % Backpropagate the errors.
            out_del = dact_fn(out_actual).*(out_vec1 - tar_vec1);
            hid_del = dact_fn(hid_input).*(out_wt'*out_del);
            
            out_wt = out_wt - learn_rate.*out_del*hid_vec_out';
            hid_wt = hid_wt - learn_rate.*hid_del*input_vec';
        end
        
        % Calculate the error for plotting.
        error = 0;
        for k = 1: batch_size
            input_vec = images(:, n(k));
            tar_vec1 = labels1(:, n(k));
            
            error = error + norm(act_fn(out_wt*act_fn(hid_wt*input_vec)) - tar_vec1, 2);
        end
        error = error/batch_size;
        
       
    end
    batch_size = 100;
    epochs = 500;
    
    % Load the test set
    images = loadMNISTImages('t10k-images.idx3-ubyte');
    labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
    
    
    testSetSize = size(images, 2);
    class_errors = 0;
    correct_class = 0;
    
    
    for n = 1: testSetSize
        input_vec = images(:, n);
        out_vec1 = evaluate(act_fn, hid_wt, out_wt, input_vec);
        
        class = decision(out_vec1);
        if class == labels(n) + 1
            correct_class = correct_class + 1;
        else
            class_errors = class_errors + 1;
        end
    end
    
    accuracy = (correct_class/10000)*100;
    
    function class = decision(out_vec1)

    max = 0;
    class = 1;
    for i = 1: size(out_vec1, 1)
        if out_vec1(i) > max
            max = out_vec1(i);
            class = i;
        end
    end
end

    
    function out_vec1 = evaluate(activationFunction, hid_wt, out_wt, input_vec)

    out_vec1 = activationFunction(out_wt*activationFunction(hid_wt*input_vec));
end
    