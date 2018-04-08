// g2o - General Graph Optimization
// Copyright (C) 2011 Kurt Konolige
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef G2O_SBA_TYPES
#define G2O_SBA_TYPES

#include "../core/base_vertex.h"

#include <Eigen/Geometry>
#include <iostream>

namespace g2o {

/**
 * \brief Point vertex, XYZ
 */
    class VertexSBAPointXYZ : public BaseVertex<3, Vector3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexSBAPointXYZ();

        virtual bool read(std::istream &is);

        virtual bool write(std::ostream &os) const;

        virtual void setToOriginImpl() {
            _estimate.fill(0.);
        }

        virtual void oplusImpl(const double *update) {
            Eigen::Map<const Vector3d> v(update);
            _estimate += v;
        }
    };

    class VertexDepthScale : public BaseVertex<2, Vector2d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexDepthScale();

        virtual bool read(std::istream &is);

        virtual bool write(std::ostream &os) const;

        virtual void setToOriginImpl() {
            _estimate.fill(0.);
        }

        virtual void oplusImpl(const double *update) {
            Eigen::Map<const Vector2d> v(update);
            _estimate += v;
        }
    };

    class VertexDepthVector : public  BaseVertex<2, Vector2d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexDepthVector();

        virtual bool read(std::istream &is);

        virtual bool write(std::ostream &os) const;

        virtual void setToOriginImpl() {
            _estimate.fill(0.);
        }

        virtual void oplusImpl(const double *update) {
            Eigen::Map<const Vector2d> v(update);
            _estimate += v;
        }
    };


    class VertexSBAPlane : public BaseVertex<4, Vector4d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexSBAPlane() : BaseVertex<4, Vector4d>() {}

        virtual bool read(std::istream &is) override { return false; }

        virtual bool write(std::ostream &os) const override { return true; }

        virtual void setToOriginImpl() {
            _estimate.fill(0);
        }

        virtual void oplusImpl(const double *update) {
            Eigen::Map<const Vector4d> v(update);
            _estimate += v;
        }
    };

    typedef Eigen::Matrix<double, 1, 9> Vector9d;
    class VertexSBAHomograph : public BaseVertex<9, Vector9d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexSBAHomograph() : BaseVertex<9, Vector9d>() {}

        virtual bool read(std::istream &is) override { return false; }

        virtual bool write(std::ostream &os) const override { return true; }

        virtual void setToOriginImpl() {
            _estimate.fill(0);
        }

        virtual void oplusImpl(const double *update) {
            Eigen::Map<const Vector9d> v(update);
            _estimate += v;
        }
    };

} // end namespace

#endif // SBA_TYPES
