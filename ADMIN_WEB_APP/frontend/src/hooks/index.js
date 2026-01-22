/**
 * CYPEARL Phase 1 Dashboard - Custom Hooks
 */

import { useState, useMemo, useCallback } from 'react';
import { DEFAULT_WEIGHTS } from '../constants';

// ============================================================================
// WEIGHTS HOOK
// ============================================================================

export const useWeights = () => {
    const [weights, setWeights] = useState({ ...DEFAULT_WEIGHTS });
    const [minClusterSize, setMinClusterSize] = useState(30);

    const weightsTotal = useMemo(() =>
        Object.values(weights).reduce((a, b) => a + b, 0),
        [weights]
    );

    const normalizedWeights = useMemo(() => {
        const total = weightsTotal || 1;
        return {
            behavioral: weights.behavioral / total,
            silhouette: weights.silhouette / total,
            stability: weights.stability / total,
            statistical: weights.statistical / total
        };
    }, [weights, weightsTotal]);

    const updateWeight = useCallback((key, value) => {
        const numValue = parseFloat(value) || 0;
        setWeights(prev => ({ ...prev, [key]: Math.max(0, Math.min(1, numValue)) }));
    }, []);

    const resetToEqual = useCallback(() => {
        setWeights({ behavioral: 0.25, silhouette: 0.25, stability: 0.25, statistical: 0.25 });
    }, []);

    const resetToDefault = useCallback(() => {
        setWeights({ ...DEFAULT_WEIGHTS });
    }, []);

    return {
        weights,
        setWeights,
        weightsTotal,
        normalizedWeights,
        updateWeight,
        resetToEqual,
        resetToDefault,
        minClusterSize,
        setMinClusterSize
    };
};

// ============================================================================
// CLUSTERING CONFIG HOOK
// ============================================================================

export const useClusteringConfig = () => {
    const [config, setConfig] = useState({
        algorithm: 'kmeans',
        k: 5,
        use_pca: true,
        pca_variance: 0.90,
        random_state: 42
    });

    const [optConfig, setOptConfig] = useState({
        algorithm: 'all',
        k_min: 2,
        k_max: 12,
        use_pca: true,
        pca_variance: 0.90
    });

    const updateConfig = useCallback((updates) => {
        setConfig(prev => ({ ...prev, ...updates }));
    }, []);

    const updateOptConfig = useCallback((updates) => {
        setOptConfig(prev => ({ ...prev, ...updates }));
    }, []);

    return {
        config,
        setConfig,
        updateConfig,
        optConfig,
        setOptConfig,
        updateOptConfig
    };
};

// ============================================================================
// CLUSTERING RESULTS HOOK
// ============================================================================

export const useClusteringResults = () => {
    const [clusteringResult, setClusteringResult] = useState(null);
    const [optimizationResult, setOptimizationResult] = useState(null);
    const [industryAnalysis, setIndustryAnalysis] = useState(null);
    const [personaLabels, setPersonaLabels] = useState({});

    const clearResults = useCallback(() => {
        setClusteringResult(null);
        setOptimizationResult(null);
        setIndustryAnalysis(null);
    }, []);

    const updatePersonaLabel = useCallback((clusterId, label) => {
        setPersonaLabels(prev => ({ ...prev, [clusterId]: label }));
    }, []);

    return {
        clusteringResult,
        setClusteringResult,
        optimizationResult,
        setOptimizationResult,
        industryAnalysis,
        setIndustryAnalysis,
        personaLabels,
        setPersonaLabels,
        updatePersonaLabel,
        clearResults
    };
};