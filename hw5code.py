import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector, min_samples_leaf=None):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    arg = np.argsort(feature_vector)
    feature_vector = feature_vector[arg]
    target_vector = target_vector[arg]
    thresholds = (feature_vector + np.roll(feature_vector, 1))[1:] / 2
    
    left_objects_mask = feature_vector[np.newaxis, :] < thresholds[:, np.newaxis]
    right_objects_mask = ~left_objects_mask
    if min_samples_leaf == None:
        non_zero_lines = left_objects_mask.any(axis=1) & right_objects_mask.any(axis=1)
        thresholds = thresholds[non_zero_lines]
    if not min_samples_leaf == None:
        min_samples_leaf_mask = ((np.sum(left_objects_mask, axis=1) >= min_samples_leaf)
                                 & (np.sum(left_objects_mask, axis=1) >= min_samples_leaf)
                                 & left_objects_mask.any(axis=1) 
                                 & right_objects_mask.any(axis=1))
        thresholds = thresholds[min_samples_leaf_mask]
    if np.size(thresholds) == 0:
        return None, None, None, None
    left_objects_mask = feature_vector[np.newaxis, :] < thresholds[:, np.newaxis]
    right_objects_mask = ~left_objects_mask
 
    proportions_of_left_subsamples = np.mean(left_objects_mask, axis=1)
    proportions_of_right_subsamples = 1 - proportions_of_left_subsamples
 
    pos_class_mask = target_vector[np.newaxis, :]
    neg_class_mask = ~pos_class_mask
    p_0_left = np.sum(left_objects_mask & neg_class_mask, axis=1) / np.sum(left_objects_mask, axis=1)
    p_1_left = np.sum(left_objects_mask & pos_class_mask, axis=1) / np.sum(left_objects_mask, axis=1)
    p_0_right = np.sum(right_objects_mask & neg_class_mask, axis=1) / np.sum(right_objects_mask, axis=1)
    p_1_right = np.sum(right_objects_mask & pos_class_mask, axis=1) / np.sum(right_objects_mask, axis=1)
 
    left_ginis = -proportions_of_left_subsamples * (1 - p_0_left**2 - p_1_left**2)
    right_ginis = -proportions_of_right_subsamples * (1 - p_0_right**2 - p_1_right**2)
    
    ginis = left_ginis + right_ginis
    pos_best_gini = np.argmax(ginis)
    threshold_best = thresholds[pos_best_gini]
    gini_best = ginis[pos_best_gini]
    return thresholds, ginis, threshold_best, gini_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        ####
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        
    
    def get_params(self, deep=False):
        # added
        return {'feature_types': self._feature_types, 
               'max_depth': self._max_depth, 
               'min_samples_split': self._min_samples_split,
               'min_samples_leaf': self._min_samples_leaf}
        

    def _fit_node(self, sub_X, sub_y, node):
        
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        self.depth += 1
        
        if not self._max_depth == None:
            if self.depth >= self._max_depth:
                node["type"] = "terminal"
                node['class'] = int(round(np.mean(sub_y)))
                return
        

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]): # iterate over the features in the feature-matrix (do not exclude the first row)
            feature_type = self._feature_types[feature] # take the feature type from the list of feature types based on index
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature] # create the feature vector
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature]) # count dictionary for all the categories
                clicks = Counter(sub_X[sub_y == 1, feature]) # count only how many instances of a category have value 1 of the target
                ratio = {}
                for key, current_count in counts.items(): # iterate over the count dictionary for all categories
                    if key in clicks: # if the category has instance that has label: y=1
                        current_click = clicks[key] # then set current_click to the number of y=1 instances of that category
                    else:
                        current_click = 0
                        # we set the ratio of y=1 for each category
                    ratio[key] = current_click / current_count  # we need a fraction of y=1 (not the other way around)
                sorted_categories = sorted(ratio.keys(), key=lambda k: ratio[k]) # get the list of sorted categories
                categories_map = dict(zip(sorted_categories, range(len(sorted_categories)))) # disctionary (category: place)

                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]]) # for every instace get its feature position
            else:
                raise ValueError

            if np.all(feature_vector == feature_vector[0]): # if we cannot split (the feature is constant)
                continue
            
            if self._min_samples_leaf == None:
                _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            else: 
                _, _, threshold, gini = find_best_split(feature_vector, sub_y, self._min_samples_leaf)
                if gini == None:
                    node["type"] = "terminal"
                    node['class'] = int(round(np.mean(sub_y)))
                    return
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(  
                            map(lambda x: x[0], 
                                filter(
                                        lambda x: x[1] < threshold,
                                        categories_map.items())))
                else:
                    raise ValueError
        # iteration is over
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}

        
        if not self._min_samples_split == None:
            if sub_X[split].shape[0] < self._min_samples_split:
                node["type"] = "terminal"
                node['class'] = int(round(np.mean(sub_y)))
                return
            
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(
                sub_X[np.logical_not(split)],
                sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        # define the funtion recursively
        if(node['type'] == 'terminal'):
            return node['class']
        else:
            feature_type = self._feature_types[node['feature_split']]
            if(feature_type == 'real'):
                if(x[node['feature_split']] < node['threshold']):
                    return self._predict_node(x, node['left_child'])
                else:
                    return self._predict_node(x, node['right_child'])
            else:
                if(x[node['feature_split']] in node['categories_split']):
                    return self._predict_node(x, node['left_child'])
                else:
                    return self._predict_node(x, node['right_child'])
                
    def fit(self, X, y):
        self.depth = 1
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
