#include <dress/embedding.h>
#include "host_embedding.h"
#include "hypra_embedding.h"
#include "naive_embedding.h"
#include "hash_embedding.h"

namespace Dress {

Embedding* Embedding::create(const EmbeddingOption& opt) {
    // TODO: Different types
    Embedding* emb;
    if (opt.type == EmbeddingOption::Host) {
        emb = new HostEmbedding(opt.item_sz, opt.ktype, opt.dtype);
    } else if (opt.type == EmbeddingOption::Hypra) {
        emb = new HypraEmbedding(opt.item_sz, opt.ktype, opt.dtype);
    } else if (opt.type == EmbeddingOption::Naive) {
        emb = new NaiveEmbedding(opt.item_sz, opt.ktype, opt.dtype);
    } else if (opt.type == EmbeddingOption::Hash) {
        emb = new HashEmbedding(opt.item_sz, opt.ktype, opt.dtype);
    }
    emb->type_ = opt.type;
    return emb;
}

};  // Dress
