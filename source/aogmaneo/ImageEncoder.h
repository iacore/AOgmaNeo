// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Helpers.h"

namespace aon {
// Image coder
class ImageEncoder {
public:
    // Visible layer descriptor
    struct VisibleLayerDesc {
        Int3 size; // Size of input

        int radius; // Radius onto input

        // Defaults
        VisibleLayerDesc()
        :
        size(4, 4, 16),
        radius(2)
        {}
    };

    // Visible layer
    struct VisibleLayer {
        FloatBuffer weights; // Byte weight matrix

        FloatBuffer reconstruction;
    };

private:
    struct FloatInt {
        float a;
        int i;

        bool operator<(
            const FloatInt &other
        ) const {
            return a < other.a;
        }
    };

    Int3 hiddenSize; // Size of hidden/output layer

    Array<FloatInt> hiddenActivations;

    IntBuffer hiddenCs; // Hidden states

    FloatBuffer hiddenResources; // Resources

    // Visible layers and associated descriptors
    Array<VisibleLayer> visibleLayers;
    Array<VisibleLayerDesc> visibleLayerDescs;
    
    // --- Kernels ---
    
    void forward(
        const Int2 &pos,
        const Array<const FloatBuffer*> &inputCs,
        bool learnEnabled
    );

    void reconstruct(
        const Int2 &pos,
        const IntBuffer* reconCs,
        int vli
    );

public:
    float alpha;
    float gamma;

    // Defaults
    ImageEncoder()
    :
    alpha(0.02f),
    gamma(1.0f)
    {}

    // Create a sparse coding layer with random initialization
    void initRandom(
        const Int3 &hiddenSize, // Hidden/output size
        const Array<VisibleLayerDesc> &visibleLayerDescs // Descriptors for visible layers
    );

    // Activate the sparse coder (perform sparse coding)
    void step(
        const Array<const FloatBuffer*> &inputs, // Input states
        bool learnEnabled // Whether to learn
    );

    void reconstruct(
        const IntBuffer* reconCs
    );

    const FloatBuffer &getReconstruction(
        int i
    ) const {
        return visibleLayers[i].reconstruction;
    }

    // Serialization
    void write(
        StreamWriter &writer
    ) const;

    void read(
        StreamReader &reader
    );

    // Get the number of visible layers
    int getNumVisibleLayers() const {
        return visibleLayers.size();
    }

    // Get a visible layer
    const VisibleLayer &getVisibleLayer(
        int i // Index of visible layer
    ) const {
        return visibleLayers[i];
    }

    // Get a visible layer descriptor
    const VisibleLayerDesc &getVisibleLayerDesc(
        int i // Index of visible layer
    ) const {
        return visibleLayerDescs[i];
    }

    // Get the hidden states
    const IntBuffer &getHiddenCs() const {
        return hiddenCs;
    }

    // Get the hidden size
    const Int3 &getHiddenSize() const {
        return hiddenSize;
    }
};
} // namespace aon
