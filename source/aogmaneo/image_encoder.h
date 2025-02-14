// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "helpers.h"

namespace aon {
// image coder
class Image_Encoder {
public:
    // visible layer descriptor
    struct Visible_Layer_Desc {
        Int3 size; // size of input

        int radius; // radius onto input

        // defaults
        Visible_Layer_Desc()
        :
        size(4, 4, 16),
        radius(2)
        {}
    };

    // visible layer
    struct Visible_Layer {
        Byte_Buffer protos;
        Byte_Buffer weights; // for reconstruction

        Byte_Buffer reconstruction;
    };

    struct Params {
        float threshold; // early stopping threshold distance
        float scale; // scale of reconstruction
        float falloff; // amount less when not maximal (multiplier)
        float lr; // learning rate
        float rr; // reconstruction rate
        
        Params()
        :
        threshold(0.001f),
        scale(2.0f),
        falloff(0.99f),
        lr(0.1f),
        rr(0.1f)
        {}
    };


    Int3 hidden_size; // size of hidden/output layer

    Int_Buffer hidden_cis; // hidden states

    Float_Buffer hidden_resources;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;
    
    // --- kernels ---
    
    void forward(
        const Int2 &column_pos,
        const Array<Byte_Buffer_View> &inputs,
        bool learn_enabled,
        unsigned long* state
    );

    void learn_reconstruction(
        const Int2 &column_pos,
        Byte_Buffer_View inputs,
        int vli,
        unsigned long* state
    );

    void reconstruct(
        const Int2 &column_pos,
        Int_Buffer_View recon_cis,
        int vli
    );

public:
    Params params;

    void init_random(
        const Int3 &hidden_size, // hidden/output size
        const Array<Visible_Layer_Desc> &visible_layer_descs // descriptors for visible layers
    );

    // activate the sparse coder (perform sparse coding)
    void step(
        const Array<Byte_Buffer_View> &inputs, // input states
        bool learn_enabled // whether to learn
    );

    void reconstruct(
        Int_Buffer_View recon_cis
    );

    const Byte_Buffer &get_reconstruction(
        int vli
    ) const {
        return visible_layers[vli].reconstruction;
    }

    // serialization
    int size() const; // returns size in bytes

    void write(
        Stream_Writer &writer
    ) const;

    void read(
        Stream_Reader &reader
    );
};
}
