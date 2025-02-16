#include <infiniband/verbs.h>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <cstring>
#include <cctype>

// Constants
#define BUCKET_SIZE 32767

// Word record structure
struct WordRecord {
    char word[64];         // Word (null-terminated)
    int64_t freq;          // Word frequency
    int32_t locations[7];  // Up to 7 locations

    // Insert a new location while keeping the array sorted
    void insert_location(int location) {
    // Check if the array already contains the maximum allowed locations
    auto it = std::find(locations, std::end(locations), 0); // Find the first uninitialized slot
    if (it != std::end(locations)) {
        *it = location; // Add location to the first available slot
        std::sort(locations, it + 1); // Keep the array sorted up to the current size
    }
    // If all slots are already filled, ignore new locations
}
};

// Distributed hash table structure
struct HashTable {
    std::vector<std::vector<WordRecord>> buckets;

    HashTable() {
        buckets.resize(256); // 256 buckets
        for (auto &bucket : buckets) {
            bucket.reserve(BUCKET_SIZE);
        }
    }

    void insert_or_update(const std::string &word, int location) {
        int bucket_index = hash_to_bucket(word);
        auto &bucket = buckets[bucket_index];
        for (auto &record : bucket) {
            if (strcmp(record.word, word.c_str()) == 0) {
                record.freq++;
                record.insert_location(location);
                return;
            }
        }

        if (bucket.size() < BUCKET_SIZE) {
            WordRecord new_record = {};
            strncpy(new_record.word, word.c_str(), 63);
            new_record.word[63] = '\0';
            new_record.freq = 1;
            std::fill(std::begin(new_record.locations), std::end(new_record.locations), 0);
            new_record.insert_location(location);
            bucket.push_back(new_record);
        }
    }

    int hash_to_bucket(const std::string &word) const {
        uint32_t hash_value = 0;
        for (size_t i = 0; i < word.size(); ++i) {
            int char_value = static_cast<int>(word[i]);
            hash_value += char_value * ((i % 2 == 0) ? 121 : 1331);
        }
        hash_value %= 2048;
        return hash_value & 0xFF; // Last 8 bits for bucket index
    }

    WordRecord find_word(const std::string &word) const {
        int bucket_index = hash_to_bucket(word);
        const auto &bucket = buckets[bucket_index];
        for (const auto &record : bucket) {
            if (strcmp(record.word, word.c_str()) == 0) {
                return record;
            }
        }
        return {};  // Return an empty record if the word is not found
    }
};

//Compute the hash value for each word
int compute_hash(const std::string &word, int &owner_process, int &bucket_index, int num_procs) {
    uint32_t hash_value = 0;
    for (size_t i = 0; i < word.size(); ++i) {
        int char_value = static_cast<int>(word[i]);
        hash_value += char_value * ((i % 2 == 0) ? 121 : 1331);
    }
    hash_value %= 2048;

    owner_process = (hash_value >> 8) & 0b111; // First 3 bits for process ID
    bucket_index = hash_value & 0xFF;         // Last 8 bits for bucket index

    return hash_value;
}

//Read the query file to get list of words
std::vector<std::string> read_file_to_words(const std::string &filename) {
    std::ifstream infile(filename);
    std::vector<std::string> words;
    std::string word;
    while (infile >> word) {
        std::string clean_word;
        for (char c : word) {
            if (isalnum(c)) {
                clean_word += std::toupper(c);
            } else if (!clean_word.empty()) {
                words.push_back(clean_word);
                clean_word.clear();
            }
        }
        if (!clean_word.empty()) {
            words.push_back(clean_word);
        }
    }
    return words;
}

//Read and process the words to fill into distributed hash table
void read_file_and_distribute(const std::string &filename, 
                              HashTable &local_table, 
                              int rank, int num_procs) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        if (rank == 0) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
        }
        return;
    }

    std::string clean_word;
    size_t char_index = 0;           // Tracks the current character index in the file
    size_t word_start_index = 0;     // Tracks the starting index of the current word

    char c;
    while (infile.get(c)) {
        if (isalnum(c)) {
            // Start a new word if clean_word is empty
            if (clean_word.empty()) {
                word_start_index = char_index;
            }
            clean_word += std::toupper(c); // Accumulate valid characters
        } else {
            // Process the word when a delimiter is encountered
            if (!clean_word.empty()) {
                int owner_process, bucket_index;
                compute_hash(clean_word, owner_process, bucket_index, num_procs);
                if (owner_process == rank) {
                    local_table.insert_or_update(clean_word, word_start_index);
                }
                clean_word.clear(); // Reset for the next word
            }
        }
        ++char_index; // Increment the character index for every character read
    }

    // Handle the last word (in case the file does not end with a delimiter)
    if (!clean_word.empty()) {
        int owner_process, bucket_index;
        compute_hash(clean_word, owner_process, bucket_index, num_procs);
        if (owner_process == rank) {
            local_table.insert_or_update(clean_word, word_start_index);
        }
    }
}

//Distribute words into hash table by computing hash
void distribute_words(const std::vector<std::string> &words, HashTable &local_table, int rank, int num_procs) {
    for (size_t i = 0; i < words.size(); ++i) {
        const auto &word = words[i];
        int owner_process, bucket_index;
        compute_hash(word, owner_process, bucket_index, num_procs);

        if (owner_process == rank) {
            local_table.insert_or_update(word, i);
        }
    }
}

//Query words from query file
void query_words(const std::vector<std::string> &queries, const HashTable &local_table, int rank, int num_procs) {
    std::vector<WordRecord> query_results(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        const auto &query = queries[i];
        int owner_process, bucket_index;
        compute_hash(query, owner_process, bucket_index, num_procs);

        WordRecord result = {};
        if (owner_process == rank) {
            result = local_table.find_word(query);
        }

        MPI_Bcast(&result, sizeof(WordRecord), MPI_BYTE, owner_process, MPI_COMM_WORLD);
        query_results[i] = result;
    }

    // Rank 0 handles output and avoids repeated words
    if (rank == 0) {
        std::cout << "\n====== ====== ====== ======\n   Starting the query ... \n====== ====== ====== ======\n";
        std::unordered_set<std::string> printed_words;
        for (size_t i = 0; i < queries.size(); ++i) {
            const auto &query = queries[i];
            if (printed_words.find(query) != printed_words.end()) {
                continue;  // Skip if already printed
            }
            printed_words.insert(query);

            const auto &record = query_results[i];
            std::cout << query << " - Freq: " << record.freq;
            if (record.freq > 0) {
                std::cout << "; Loc (<= 7): ";
                for (int loc : record.locations) {
                    if (loc > 0) std::cout << loc << " ";
                }
            }
            std::cout << std::endl;
        }
    }
}

//Outputs most frequent word for each process
void output_most_frequent_word(const HashTable &local_table, int rank) {
    WordRecord most_frequent = {};
    for (const auto &bucket : local_table.buckets) {
        for (const auto &record : bucket) {
            if (record.freq > most_frequent.freq) {
                most_frequent = record;
            }
        }
    }
    if (most_frequent.freq > 0) {
        std::cout << "Rank " << rank << ": " << most_frequent.word << " - Freq: " << most_frequent.freq
                  << "; Loc (<= 7): ";
        for (int loc : most_frequent.locations) {
            if (loc > 0) std::cout << loc << " ";
        }
        std::cout << std::endl;
    }
}

// RDMA structures
struct RDMAContext {
    struct ibv_context *context;
    struct ibv_pd *protection_domain;
    struct ibv_mr *memory_region;
    struct ibv_cq *completion_queue;
    struct ibv_qp *queue_pair;
    HashTable *hash_table;
    uint32_t rkey;
    uint64_t remote_addr;
    void *buffer;
    size_t buffer_size;
};

// Initialize RDMA context
void setup_rdma(RDMAContext &rdma_ctx, size_t buffer_size) {
    rdma_ctx.context = nullptr;
    rdma_ctx.protection_domain = nullptr;
    rdma_ctx.completion_queue = nullptr;
    rdma_ctx.queue_pair = nullptr;
    rdma_ctx.hash_table = new HashTable();
    rdma_ctx.buffer = malloc(buffer_size);
    rdma_ctx.buffer_size = buffer_size;

    // Obtain RDMA device context
    struct ibv_device **dev_list = ibv_get_device_list(nullptr);
    if (!dev_list) {
        std::cerr << "Failed to get RDMA device list\n";
        exit(1);
    }
    rdma_ctx.context = ibv_open_device(dev_list[0]);
    ibv_free_device_list(dev_list);
    if (!rdma_ctx.context) {
        std::cerr << "Failed to open RDMA device\n";
        exit(1);
    }

    // Create protection domain
    rdma_ctx.protection_domain = ibv_alloc_pd(rdma_ctx.context);
    if (!rdma_ctx.protection_domain) {
        std::cerr << "Failed to allocate protection domain\n";
        exit(1);
    }

    // Register memory
    rdma_ctx.memory_region = ibv_reg_mr(rdma_ctx.protection_domain, rdma_ctx.buffer, buffer_size,
                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!rdma_ctx.memory_region) {
        std::cerr << "Failed to register memory region\n";
        exit(1);
    }

    // Create completion queue
    rdma_ctx.completion_queue = ibv_create_cq(rdma_ctx.context, 10, nullptr, nullptr, 0);
    if (!rdma_ctx.completion_queue) {
        std::cerr << "Failed to create completion queue\n";
        exit(1);
    }

    // Create queue pair
    struct ibv_qp_init_attr qp_init_attr = {};
    qp_init_attr.send_cq = rdma_ctx.completion_queue;
    qp_init_attr.recv_cq = rdma_ctx.completion_queue;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = 10;
    qp_init_attr.cap.max_recv_wr = 10;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;

    rdma_ctx.queue_pair = ibv_create_qp(rdma_ctx.protection_domain, &qp_init_attr);
    if (!rdma_ctx.queue_pair) {
        std::cerr << "Failed to create queue pair\n";
        exit(1);
    }
}

// Perform RDMA read
void rdma_read_word(RDMAContext &rdma_ctx, const std::string &word, int owner_process,
                    HashTable &local_table, int bucket_index) {
    if (owner_process == 0) {
        // Process P0 handles the query locally
        auto record = local_table.find_word(word);
        std::cout << "Word: " << word << " - Freq: " << record.freq << std::endl;
    } else {
        // RDMA read from remote process
        struct ibv_sge sge = {};
        struct ibv_send_wr wr = {};
        struct ibv_send_wr *bad_wr = nullptr;

        sge.addr = reinterpret_cast<uint64_t>(rdma_ctx.buffer);
        sge.length = sizeof(WordRecord);
        sge.lkey = rdma_ctx.memory_region->lkey;

        struct ibv_send_wr wr = {};
        wr.wr_id = 0;
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_READ;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey = rkey;

        struct ibv_send_wr *bad_wr = nullptr;  // Corrected to match ibv_post_send's requirements
        if (ibv_post_send(ctx.queue_pair, &wr, &bad_wr)) {  // Pass address of bad_wr
            throw std::runtime_error("Failed to post RDMA read");
        }

        struct ibv_wc wc;
        while (ibv_poll_cq(ctx.completion_queue, 1, &wc) < 1) {}
        if (wc.status != IBV_WC_SUCCESS) {
            throw std::runtime_error("RDMA read failed");
        }

    }
}

// Main program
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    RDMAContext rdma_ctx = {};
    setup_rdma(rdma_ctx, 64 * 1024); // 64KB buffer

    // Remaining program logic: distribute words, populate hash tables, and query...
    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <query_file> <data_file>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string query_file = argv[1];
    std::string data_file = argv[2];

    // Step 1: Read data and distribute words
    // std::vector<std::string> data_words = read_file_to_words(data_file);
    HashTable local_table;
    // Read and distribute words directly into the local hash table
    read_file_and_distribute(data_file, local_table, rank, num_procs);
    rdma_read_word(rdma_ctx, remote_addr, ctx.rkey, &result, sizeof(WordRecord));

    // Step 2: Output the most frequent word for each local table
    MPI_Barrier(MPI_COMM_WORLD);
    output_most_frequent_word(local_table, rank);

    // Step 3: Query words from query.txt
    std::vector<std::string> query_words_list = read_file_to_words(query_file);
    MPI_Barrier(MPI_COMM_WORLD);
    query_words(query_words_list, local_table, rank, num_procs);

    // Cleanup RDMA resources
    ibv_dereg_mr(rdma_ctx.memory_region);
    ibv_dealloc_pd(rdma_ctx.protection_domain);
    ibv_close_device(rdma_ctx.context);
    free(rdma_ctx.buffer);

    MPI_Finalize();
    return 0;
}